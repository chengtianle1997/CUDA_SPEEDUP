#include "iostream"
#include "cuda_runtime.h"
#include "cuda.h"
#include "cublas.h"
#include "cublas_api.h"
#include "pch.h"
//#include "stdafx.h"
#include "cv.h"
//#include <process.h>
//#include "CameraApi.h"
#include "LaserRange.h"
//#include "afxwin.h"
//#include "windows.h"
#include "math.h"
//#include "cstdlib"
//#include "sstream"
//#include "ImProcess.h"
//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
#include "CudaTest.h"
#include <device_launch_parameters.h>
#include <device_functions.h>

//#include <stdio.h>
//#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
//#include "CudaKernal.cuh"

using namespace std;
using namespace cv;
using namespace cv::cuda;

//extern "C" void GetGaussFitCuda(GpuMat gpuMat, MPoint *point, double maxError, double minError, int yRange, int Colonce);

//矩阵乘法 a[M][N]*b[N][S]
cudaError_t checkCudaError(cudaError_t CudaFunction, const char* ident) {
	cudaError_t err = CudaFunction;
	if (err != cudaSuccess) {
		fprintf(stderr, "%s \t cudaError:%s\n", ident, cudaGetErrorString(cudaGetLastError()));
	}
	return err;
}

//Coloncel行扫描得点存储
__global__ void GetGaussPointCuda(PtrStepSz<uchar1> src, MPoint *point, int **gpu_data, int Colonce, int Rows, int Cols) {
	int threadId = threadIdx.x;
	//printf("%d\n",threadId);
	//__shared__ int *gpu_cr;
	//gpu_cr = new int [Rows*Cols];
	//逐行存入数组
	for (int j = 0; j < Rows; j++)
	{
		for (int i = 0; i < Colonce; i++)
		{
			gpu_data[i + threadId * Colonce][j] = (int)src(j, threadId*Colonce + i).x;
		}
	}
	//int i = 0, j = 0;
	//gpu_cr[i*Rows + j] = *((int*)&src( j, threadId*Colonce + i));

	//取每列最大值位置
	for (int i = 0; i < Colonce; i++) {
		int MaxPixel = gpu_data[i + threadId * Colonce][0];
		//printf("the first pixel is %d \n", MaxPixel);
		int MaxY = 0;
		for (int j = 1; j < Rows; j++)
		{
			if (gpu_data[i + threadId * Colonce][j] > MaxPixel)
			{
				MaxPixel = gpu_data[i + threadId * Colonce][j];
				MaxY = j;
			}
		}

		point[threadId*Colonce + i].x = threadId * Colonce + i;
		point[threadId*Colonce + i].y = MaxY;
		point[threadId*Colonce + i].bright = MaxPixel;
	}
	__syncthreads();
}

//按列筛选并处理高斯点
__global__ void GetGaussFitRes(MPoint *point, MatrixUnion *gpu_mar, int **gpu_data, double maxError, double minError, int yRange, int Rows, int Cols, int Precision)
{
	//通过块并行解决一个block内thread不够用的问题
	int threadId = blockIdx.x*blockDim.x + threadIdx.x;
	//判断以确定该线程有可处理数据
	if (threadId < Cols)
	{
		////高斯点存储申请
		//int *y; //存储高斯点在每列的行位置
		//int *br; //存储高斯点的值
		GPoint *gpoint = new GPoint[2 * yRange];
		int Pixnum = 0; //统计高斯点个数
		//确定上下界位置 减少计算次数
		double minLine = minError * point[threadId].bright;
		double maxLine = (1 - maxError) * point[threadId].bright;
		//高斯点筛选
		for (int i = (point[threadId].y - yRange); i < (point[threadId].y + yRange + 1); i++)
		{
			if ((gpu_data[threadId][i] > minLine) && (gpu_data[threadId][i] < maxLine))
			{
				gpoint[Pixnum].x = i;
				gpoint[Pixnum].brightness = gpu_data[threadId][i];
				Pixnum++;
			}
		}
		point[threadId].Pixnum = Pixnum;
		//__shared__ MatrixUnion *gpu_mar;

		//高斯点大于3时进行拟合
		if (Pixnum > 3) {
			//运算矩阵申请
			int n = Pixnum;

			//X矩阵（1 x x^2）  n*3
			gpu_mar[threadId].X = new long long int *[n];
			for (int i = 0; i < n; i++) {
				gpu_mar[threadId].X[i] = new long long int[3];
			}
			//XT矩阵 X的转置  3*n
			gpu_mar[threadId].XT = new long long int *[3];
			for (int i = 0; i < 3; i++) {
				gpu_mar[threadId].XT[i] = new long long int[n];
			}
			//Z矩阵(brightness) n*1
			gpu_mar[threadId].Z = new long long int[n];
			//B矩阵（结果） 3*1
			gpu_mar[threadId].B = new long long int[3];
			//SA矩阵 （XT*X） 3*3
			gpu_mar[threadId].SA = new long long int *[3];
			for (int i = 0; i < 3; i++) {
				gpu_mar[threadId].SA[i] = new long long int[3];
			}
			//SAN矩阵 SA的逆矩阵  3*3
			gpu_mar[threadId].SAN = new float *[3];
			for (int i = 0; i < 3; i++) {
				gpu_mar[threadId].SAN[i] = new float[3];
			}
			//SC矩阵  SAN*XT 3*n
			gpu_mar[threadId].SC = new float *[3];
			for (int i = 0; i < 3; i++) {
				gpu_mar[threadId].SC[i] = new float[n];
			}


			/*
			//X矩阵（1 x x^2）  n*3
			double **X = new double*[n];
			for (int i = 0; i < n; i++) {
				X[i] = new double[3];
			}

			//XT矩阵 X的转置  3*n
			double **XT = new double*[3];
			for (int i = 0; i < 3; i++) {
				XT[i] = new double[n];
			}

			//Z矩阵(brightness) n*1
			double *Z = new double[n];
			//B矩阵（结果） 3*1
			double *B = new double[3];

			//SA矩阵 （XT*X） 3*3
			double **SA = new double*[3];
			for (int i = 0; i < 3; i++) {
				SA[i] = new double[3];
			}

			//SAN矩阵 SA的逆矩阵  3*3
			double **SAN = new double*[3];
			for (int i = 0; i < 3; i++) {
				SAN[i] = new double[3];
			}
			*/

			//存入X矩阵和Z矩阵 顺手存入转置XT
			for (int i = 0; i < n; i++) {
				gpu_mar[threadId].X[i][0] = 1 * Precision;
				gpu_mar[threadId].X[i][1] = gpoint[i].x*Precision;
				gpu_mar[threadId].X[i][2] = gpoint[i].x*gpoint[i].x*Precision;
				gpu_mar[threadId].Z[i] = gpoint[i].brightness*Precision;
				gpu_mar[threadId].XT[0][i] = 1 * Precision;
				gpu_mar[threadId].XT[1][i] = gpoint[i].x*Precision;
				gpu_mar[threadId].XT[2][i] = gpoint[i].x*gpoint[i].x*Precision;
			}
			/*for (int i_ = 0; i_ < n; i_++)
			{
				for (int j_ = 0; j_ < 3; j_++)
				{
					printf("%d\t", gpu_mar[threadId].X[i_][j_]);
				}
				printf("\n");

			}*/
			//计算SA = XT*X
			for (int m = 0; m < 3; m++) {
				for (int s = 0; s < 3; s++) {
					gpu_mar[threadId].SA[m][s] = 0;
					for (int i = 0; i < n; i++) {
						gpu_mar[threadId].SA[m][s] += gpu_mar[threadId].XT[m][i] * gpu_mar[threadId].X[i][s];
					}
				}
			}

			/*if (threadId == 20) {
				for (int i_ = 0; i_ < 3; i_++)
				{
					for (int j_ = 0; j_ < 3; j_++)
					{
						printf("%lld\t", gpu_mar[threadId].SA[i_][j_]);
					}
					printf("\n");
				}
			}*/

			//计算SAN
			const int mat_num = 3;//求逆矩阵的阶数
			gpu_mar[threadId].in_v = new float*[mat_num];
			float **in_v = gpu_mar[threadId].in_v;
			for (int i_ = 0; i_ < mat_num; i_++)
			{
				in_v[i_] = new float[mat_num];
			}
			gpu_mar[threadId].BC = new float*[mat_num];
			float **BC = gpu_mar[threadId].BC;

			for (int i_ = 0; i_ < mat_num; i_++)
			{
				BC[i_] = new float[mat_num];
			}

			for (int i_ = 0; i_ < mat_num; i_++)
			{
				BC[i_][i_] = 1;
				for (int j_ = 0; j_ < mat_num; j_++)
				{
					if (j_ == i_)
						continue;
					BC[i_][j_] = 0;
				}
			}

			for (int i = 0; i < mat_num; i++)
			{
				//L矩阵建立
				gpu_mar[threadId].L = new float*[mat_num];
				float **L = gpu_mar[threadId].L;
				for (int i_ = 0; i_ < mat_num; i_++)
				{
					L[i_] = new float[mat_num];
				}
				//U矩阵建立
				gpu_mar[threadId].U = new float*[mat_num];
				float **U = gpu_mar[threadId].U;
				for (int i_ = 0; i_ < mat_num; i_++)
				{
					U[i_] = new float[mat_num];
				}

				gpu_mar[threadId].P = new int[mat_num];
				int *P = gpu_mar[threadId].P;
				//SA复制
				gpu_mar[threadId].A_mirror = new long long int*[mat_num];
				long long int **A_mirror = gpu_mar[threadId].A_mirror;
				for (int i_ = 0; i_ < mat_num; i_++)
				{
					A_mirror[i_] = new long long int[mat_num];
				}
				for (int i_ = 0; i_ < mat_num; i_++)
				{
					for (int j_ = 0; j_ < mat_num; j_++)
					{
						A_mirror[i_][j_] = gpu_mar[threadId].SA[i_][j_];
					}
				}


				//核心part  并没有看懂
				int  row = 0;
				for (int i_ = 0; i_ < mat_num; i_++)
					P[i_] = i_;
				for (int i_ = 0; i_ < mat_num - 1; i_++)
				{
					int p = 0;
					for (int j_ = i_; j_ < mat_num; j_++)
					{
						if (A_mirror[j_][i_] > p || (-1)* A_mirror[j_][i_] > p)
						{
							p = A_mirror[j_][i_] > 0 ? A_mirror[j_][i_] : (-1)*A_mirror[j_][i_];
							row = j_;
						}
					}

					/*
					if (p == 0)
					{
						;
					}
					*/

					//交换P[i_] P[row]
					int tmp = P[i_];
					P[i_] = P[row];
					P[row] = tmp;

					//交换A[i_][j_] 和A[row][j_]
					float tmp2 = 0;
					for (int j_ = 0; j_ < mat_num; j_++)
					{
						tmp2 = A_mirror[i_][j_];
						A_mirror[i_][j_] = A_mirror[row][j_];
						A_mirror[row][j_] = tmp2;
					}

					float u = A_mirror[i_][i_], l = 0;
					for (int j_ = i_ + 1; j_ < mat_num; j_++)
					{
						l = A_mirror[j_][i_] / u;
						A_mirror[j_][i_] = l;
						for (int k_ = i_ + 1; k_ < mat_num; k_++)
						{
							A_mirror[j_][k_] = A_mirror[j_][k_] - A_mirror[i_][k_] * l;
						}

					}

				}

				//构造LU
				for (int i_ = 0; i_ < mat_num; i_++)
				{
					for (int j_ = 0; j_ <= i_; j_++)
					{
						if (i_ != j_)
							L[i_][j_] = A_mirror[i_][j_];
						else
							L[i_][j_] = 1;
					}
					for (int k_ = i_; k_ < mat_num; k_++)
					{
						U[i_][k_] = A_mirror[i_][k_];
					}
				}

				float *y = new float[mat_num];
				//正向替换
				for (int i_ = 0; i_ < mat_num; i_++)
				{
					y[i_] = BC[i][P[i_]];
					for (int j_ = 0; j_ < i_; j_++)
					{
						y[i_] = y[i_] - L[i_][j_] * y[j_];
					}
				}
				//反向替换
				for (int i_ = mat_num - 1; i_ >= 0; i_--)
				{
					in_v[i][i_] = y[i_];
					for (int j_ = mat_num - 1; j_ > i_; j_--)
					{
						in_v[i][i_] = in_v[i][i_] - U[i_][j_] * in_v[i][j_];
					}
					in_v[i][i_] /= U[i_][i_];
				}


				for (int i_ = 0; i_ < mat_num; i_++)
					delete[] L[i_];
				delete[]L;
				for (int i_ = 0; i_ < mat_num; i_++)
					delete[]U[i_];
				delete[]U;
				for (int i_ = 0; i_ < mat_num; i_++)
					delete[]A_mirror[i_];
				delete[]A_mirror;

				delete[]P;
				delete[]y;


			}

			for (int i_ = 0; i_ < mat_num; i_++)
			{
				for (int j_ = 0; j_ < i_; j_++)
				{
					gpu_mar[threadId].SAN[i_][j_] = in_v[j_][i_];
					gpu_mar[threadId].SAN[j_][i_] = in_v[i_][j_];
				}
				gpu_mar[threadId].SAN[i_][i_] = in_v[i_][i_];
			}

			for (int i = 0; i < mat_num; i++)
				delete[]in_v[i];
			delete[]in_v;
			for (int i = 0; i < mat_num; i++)
				delete[]BC[i];
			delete[]BC;
			/*
						if (threadId == 20) {
							for (int i_ = 0; i_ < mat_num; i_++)
							{
								for (int j_ = 0; j_ < mat_num; j_++)
								{
									printf("%lf\t", gpu_mar[threadId].SAN[i_][j_]);
								}
								printf("\n");
							}
						}*/

						//计算SC = SAN*XT
			for (int m = 0; m < 3; m++) {
				for (int s = 0; s < n; s++) {
					gpu_mar[threadId].SC[m][s] = 0;
					for (int i = 0; i < 3; i++) {
						gpu_mar[threadId].SC[m][s] += gpu_mar[threadId].SAN[m][i] * gpu_mar[threadId].XT[i][s];
					}
				}
			}
			/*if (threadId == 20) {
				for (int i_ = 0; i_ < 3; i_++)
				{
					for (int j_ = 0; j_ < n; j_++)
					{
						printf("%lf\t", gpu_mar[threadId].SC[i_][j_]);
					}
					printf("\n");
				}
			}*/
			//计算B = SC*Z
			for (int m = 0; m < 3; m++) {
				gpu_mar[threadId].B[m] = 0;
				for (int i = 0; i < n; i++) {
					gpu_mar[threadId].B[m] += gpu_mar[threadId].SC[m][i] * gpu_mar[threadId].Z[i];
				}
			}
			//解析B
			//printf("B1 = %d , B2 = %d\n", gpu_mar[threadId].B[1], gpu_mar[threadId].B[2]);
			point[threadId].cx = threadId;
			point[threadId].cy = 1.0*(-gpu_mar[threadId].B[1]) / ((2 * gpu_mar[threadId].B[2]));

			//point[threadId].gaussbright = exp((float)(gpu_mar[threadId].B[0] - gpu_mar[threadId].B[1] * gpu_mar[threadId].B[1] / (4 * gpu_mar[threadId].B[2])));


			for (int i = 0; i < n; i++) {
				delete[] gpu_mar[threadId].X[i];
			}
			delete[] gpu_mar[threadId].X;
			for (int i = 0; i < 3; i++) {
				delete[] gpu_mar[threadId].XT[i];
			}
			delete[] gpu_mar[threadId].XT;
			delete[] gpu_mar[threadId].Z;
			delete[] gpu_mar[threadId].B;
			for (int i = 0; i < 3; i++) {
				delete[] gpu_mar[threadId].SA[i];
			}
			delete[] gpu_mar[threadId].SA;
			for (int i = 0; i < 3; i++) {
				delete[] gpu_mar[threadId].SAN[i];
			}
			delete[] gpu_mar[threadId].SAN;
			for (int i = 0; i < 3; i++) {
				delete[] gpu_mar[threadId].SC[i];
			}
			delete[] gpu_mar[threadId].SC;

		}
		else
		{
			point[threadId].cx = threadId;
			point[threadId].cy = 0;
			point[threadId].bright = 0;
		}



		delete[] gpoint;

	}


}

//高斯点筛选
//for (int i = 0; i < Colonce; i++)
//{
//	int Pixnum = 0;
//	//GPoint *gpoint;
//	//point[threadId*Colonce+i].gpoint = new GPoint[Rows];
//	//point[i].gpoint = new GPoint[Rows];
//	for (int j = 0; j < Rows; j++)
//	{
//		if ((gpu_cr[Rows*i + j] > minError*point[threadId*Colonce + i].bright)
//			&& (gpu_cr[Rows*i + j] < (1 - maxError)*point[threadId*Colonce + i].bright)
//			&& (abs(j - point[threadId*Colonce + i].y) < yRange))
//		{
//			point[threadId*Colonce + i].gpoint[Pixnum].x = threadId * Colonce + i;
//			point[threadId*Colonce + i].gpoint[Pixnum].brightness = gpu_cr[Rows*i + j];
//			Pixnum++;
//		}
//		if ((j - point[threadId*Colonce + i].y) < yRange)
//			break;
//	}
//	point[threadId*Colonce + i].Pixnum = Pixnum;

	/*
	//矩阵运算
	if (Pixnum >= 3)
	{
		__shared__ int *X;
		X = new int[Pixnum * 3];
		__shared__ int *Z;
		Z = new int[Pixnum];
		//计算能力<3.5 不能嵌套并行核函数
		//dim3 blockSEX(1, 0, 0);
		//dim3 threadSEX(Pixnum, 0, 0);
		//存入X、Z矩阵
		//SetElementX << <blockSEX, threadSEX >> > (gpoint, X, Pixnum);
		//存入X矩阵(n*3) Z矩阵（n*1)
		for (int i = 0; i < Pixnum; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				if (j = 0)
				{
					X[i * 3 + j] = 1;
				}
				if (j = 1)
				{
					X[i * 3 + j] = gpoint[i].x;
				}
				if (j = 2)
				{
					X[i * 3 + j] = gpoint[i].x*gpoint[i].x;
				}
			}
			Z[i] = gpoint[i].brightness;
		}
		//求X转置
		__shared__ int *XT;
		XT = new int[Pixnum* 3];
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < Pixnum; j++)
			{
				XT[i*Pixnum + j] = X[j * 3 + i];
			}
		}
		//求XT*X结果
		__shared__ int *SA;
		SA = new int[3 * 3];
		for (int m = 0; i < 3; i++)
		{
			for (int s = 0; s < 3; s++)
			{
				for (int n = 0; n < Pixnum; n++)
				{
					SA[m * 3 + s] = XT[m*Pixnum + n] * X[n * 3 + s];
				}
			}
		}
		//求SA逆矩阵
		__shared__ int *SAN;
		SAN = new int[3 * 3];


	}*/
	//}
	//delete &gpu_cr;


//#define N 3
//__global__ void MatAdd(const int **A, const int **B, int **C)
//{
//	int i = threadIdx.x;
//	int j = threadIdx.y;
//	C[i][j] = A[i][j] + B[i][j];
//	//__syncthreads();
//}



//extern "C" void GetGaussFitCuda(GpuMat gpuMat, MPoint *point, double maxError, double minError, int yRange, int Colonce);

extern "C"
void CudaGuassHC(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce, int Precision) {

	int Rows = matImage.rows;
	int Cols = matImage.cols;// *matImage.channels();
	//InputArray inputMat(matImage);
	//for (int j = 0; j < Rows; j++) {
	//	//uchar* data = gpuMat.ptr<uchar>(j);
	//	for (int i = 0; i < Cols; i++) {
	//		int datt = inputMat.ptr<uchar>(j)[i];
	//		//cout << "(" << i << "," <<j << "):" << datt << endl;
	//		printf("(%d,%d):%d\n", i, j, datt);
	//	}
	//}
	//cout << Cols << endl;
	GpuMat gpuMat(matImage);
	//gpuMat.upload(matImage);
	//for (int j = 0; j < Rows; j++) {
	//	//uchar* data = gpuMat.ptr<uchar>(j);
	//	for (int i = 0; i < Cols; i++) {
	//		int datt = gpuMat.ptr<uchar>(j)[i];
	//		//cout << "(" << i << "," <<j << "):" << datt << endl;
	//		printf("(%d,%d):%d\n", i, j, datt);
	//	}
	//}
	//结构体指针上传
	MPoint *gpu_point;
	//gpu_point = new MPoint[Cols];	
	checkCudaError(cudaMalloc((void**)&gpu_point, sizeof(MPoint)*Cols), "malloc error1");
	//显存图像缓存矩阵
	int **gpu_data;
	int *gpu_data_d;
	int **cpu_data = (int**)malloc(sizeof(int*)*Cols);
	int *cpu_data_d = (int*)malloc(sizeof(int)*Cols*Rows);
	checkCudaError(cudaMalloc((void**)&gpu_data, Cols * sizeof(int**)), "malloc error2");
	checkCudaError(cudaMalloc((void**)&gpu_data_d, Cols *Rows * sizeof(int)), " malloc error2");
	for (int i = 0; i < Cols; i++) {
		cpu_data[i] = gpu_data_d + Rows * i;
		//首地址赋值 将一维矩阵转为二维
	}
	checkCudaError(cudaMemcpy(gpu_data, cpu_data, sizeof(int*)*Cols, cudaMemcpyHostToDevice), "memcpy error1");
	checkCudaError(cudaMemcpy(gpu_data_d, cpu_data_d, sizeof(int)*Rows*Cols, cudaMemcpyHostToDevice), "memcpy error1");

	/*if (cudaSuccess != cudaMemcpy(gpu_point, point, sizeof(point)*Cols, cudaMemcpyHostToDevice)) {
		printf("cuda memcpy up error1!\n");
	}*/

	//dim3 threads_all(Cols / Colonce);
	//GPU端矩阵集
	MatrixUnion *gpu_mar;
	checkCudaError(cudaMalloc((void**)&gpu_mar, sizeof(MatrixUnion)*Cols), "malloc error3");
	//每colonce列统一存入 
	GetGaussPointCuda << <1, Cols / Colonce >> > (gpuMat, gpu_point, gpu_data, Colonce, Rows, Cols);
	cudaDeviceSynchronize();
	//规划并行流  之后设计为只规划一次
	const int BlockPMat = 1280;
	int Blocknum, Threadnum;
	Blocknum = BlockPMat;
	Threadnum = Cols / Blocknum + 1;
	/*
	if (Cols > 1024) {
		Blocknum = Cols / 1024 + 1;
		Threadnum = 1024;
	}
	else {
		Blocknum = 1;
		Threadnum = Cols;
	}
	*/
	//进行高斯拟合
	GetGaussFitRes << <Blocknum, Threadnum >> > (gpu_point, gpu_mar, gpu_data, maxError, minError, yRange, Rows, Cols, Precision);
	cudaDeviceSynchronize();
	checkCudaError(cudaMemcpy(point, gpu_point, sizeof(MPoint)*Cols, cudaMemcpyDeviceToHost), "memcpy down error1");

	for (int i = 0; i < Cols; i++)
	{
		//cout << "("<<point[i].x<<","<< point[i].y<<"):"<< point[i].bright << endl;
		printf("(%d,%d):%d\t, here are %d gausspoints, the result is （%lf,%lf）\n", point[i].x, point[i].y, point[i].bright, point[i].Pixnum, point[i].cx, point[i].cy);
	}


	/*for (int i = 0; i < Cols; i++)
	{
		free((void*)cpu_data[i]);
	}*/
	free((void*)cpu_data);
	free(cpu_data_d);
	/*for (int i = 0; i < Cols; i++)
	{
		cudaFree((void*)gpu_data[i]);
	}*/
	cudaFree(gpu_data);
	cudaFree(gpu_point);
	cudaFree(gpu_data_d);
	cudaFree(gpu_mar);
	gpuMat.release();
}

extern "C" void GuassFitGpuHcT(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce)
{

}


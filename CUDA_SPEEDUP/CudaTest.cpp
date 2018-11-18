#include "cuda.h"
#include "cuda_runtime.h"
#include "pch.h"
#include <iostream>
//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "cv.h"
#include <process.h>
#include "CameraApi.h"
#include "LaserRange.h"
#include "windows.h"
#include "math.h"
#include "cstdlib"
#include "sstream"
//#include "ImProcess.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "CudaTest.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "CudaTest.h"
//#include "CudaKernal.cuh"


#define THREAD_NUM 1024
using namespace std;
using namespace cv;
using namespace cv::cuda;



int printDeviceProp(cudaDeviceProp &prop) {
	printf("Device Name:%s.\n", prop.name);
	printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %d.\n", prop.memPitch);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem : %d.\n", prop.totalConstMem);
	printf("major.minor : %d.%d.\n", prop.major, prop.minor);
	printf("clockRate : %d.\n", prop.clockRate);
	printf("textureAlignment : %d.\n", prop.textureAlignment);
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
	return 0;
}
bool InitCuda() {
	int count;
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no cuda device!\n");
		return false;
	}
	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printDeviceProp(prop);
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}

	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting cuda!\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}

//__global__ void vectorAdd(float *x, float*y, float*z) {
//	int index = threadIdx.x;
//	if (index < blockDim.x) {
//		z[index] = x[index] + y[index];
//	}
//
//}

void GlobalTest() {
	float *x;
	float *y;
	float *z;
	//vectorAdd<<<1, 10>>>(x, y, z);
}





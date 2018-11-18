#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "pch.h"
#include <iostream>
//#include "stdafx.h"
//#include <opencv2/opencv.hpp>
#include "cv.h"
//#include <process.h>
//#include "CameraApi.h"
//#include "LaserRange.h"
//#include "windows.h"
//#include "math.h"
//#include "cstdlib"
//#include "sstream"
//#include "ImProcess.h"
//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;


//打印cuda属性信息
int printDeviceProp(const cudaDeviceProp &prop);
//cuda初始化
bool InitCuda();

//__global__ void GetGaussFitCuda(GpuMat gpuMat, MPoint *point, double maxError, double minError, int yRange, int Colonce);

extern "C"
void CudaGuassHC(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce, int Precision);
//extern "C"
//__global__ void GetGaussPointCuda(PtrStepSz<uchar1> src, MPoint *point, double maxError, double minError, int yRange, int Colonce, int Rows, int Cols,);
//extern "C" void GuassFitGpuHcT(Mat matImage, MPoint *point, double maxError, double minError, int yRange, int Colonce);

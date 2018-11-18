#pragma once
#include "iostream"
#include "stdio.h"
#include "stdint.h"
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"
#include "cxcore.h"
#include <opencv2/opencv.hpp>
#include <process.h>
//#include "afxwin.h"
#include "math.h"
#include "cstdlib"

// #include "ImProcess.h"


using namespace std;
using namespace cv;


class LaserRange
{
public:
	typedef struct RangeResult {
		unsigned int maxCol;
		unsigned int maxRow;
		unsigned int maxPixel;
		double Range;
		unsigned int  PixfromCent;
	} RangeResult;
	RangeResult *GetRange(IplImage *imgRange, IplImage *imgDst);
	LaserRange();
	virtual ~LaserRange();

private:
	unsigned int maxW;
	unsigned int maxH;
	unsigned int MaxPixel;
	RangeResult *strctResult;

	//value used for calculating range from captured image data
	const double gain;//gain constant used for converting pixel offset to angle in radians
	const double offset;//offset constant
	const double h_cm;//distance between center of camera and laser
	unsigned int pixel_from_center;//brightest pixel location from center
	void Preprocess(void *img, IplImage *imgTemp);
};

class CLaserVisionDlg
{
public:
	int CaptureImage(IplImage *iplimage);
};

typedef struct GPoint {
	int x;
	int brightness;
} GPoint;

typedef struct MPoint
{
	int x;
	int y;
	double cx;
	double cy;
	int bright;
	int Pixnum;
	double gaussbright;
	//int xstart;
	//int xstop;
	//int errorup;
	//GPoint *gpoint;
} MPoint;

typedef struct MatrixUnion
{
	long long int **X;
	long long int **XT;
	long long int *Z;
	long long int *B;
	long long int **SA;
	float **SAN;//逆矩阵
	float **SC;
	float **in_v;//逆矩阵的转置
	float **BC;
	float **L; //LU过程矩阵L
	float **U; //LU过程矩阵U
	long long int **A_mirror;  //原矩阵的复制
	int *P; //LUP 记录列坐标
}MatrixUnion;

//张正友相机标定法
void calibfirst(Mat matImage);



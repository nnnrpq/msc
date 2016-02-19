#ifndef I_PP_H_INCLUDED //I_PP referes to Image_PreProcessing
#define I_PP_H_INCLUDED
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include<stdio.h>
#include <vector>
using namespace cv;
using namespace std;
Mat CannyThreshold(Mat, Mat, Mat);
Mat CannyThreshold_MemoryImages(Mat);
Mat resize_image(Mat , uint64_t, double * );
Mat padImageMatrix(Mat ,int ,int );
int fullSizeCount(Mat);
Mat foregroundBackgroundImageChange(Mat );
#endif

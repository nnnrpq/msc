#pragma once
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;


Mat Rand_Transform(Mat src, double & theta, double & xtranslate, double & ytranslate, double & scale,int flag=1);
Mat Combine_Transform(Mat t1, Mat t2);
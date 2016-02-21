//
//  Learn_Image.h
//  
//
//  Created by Rohit Shukla on 7/21/15.
//
//

#ifndef _Learn_Image_h
#define _Learn_Image_h

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include<stdio.h>
#include<vector>
using namespace cv;
using namespace std;
double Verify_Object(Mat, Mat, double);
Mat Learn_New_Transformation(Mat, Mat, vector<int>);
int UpdateLayers(int);
Mat Image_Match(Mat, Mat);
Mat MyfindAffine(vector<Point2f> obj, vector<Point2f> scene, int type, int maxtrail = 10000);
void mynormalize(vector<Point2f> src, vector<Point3f> &dst, Mat &T);
#endif

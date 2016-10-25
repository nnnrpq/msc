#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "Image_PreProcessing.h"
#include <algorithm>
#include <vector>
#include <iostream>
#include "region_of_interest.h"
using namespace cv;
using namespace std;
/// Global variables

/* The preprocessing part needs to be redesigned
in order to meet the requirement of real world image
Zhangyuan 06/12/2016*/


Mat orientationMap(const cv::Mat& mag, const cv::Mat& ori, double thresh = 1.0)
{
    Mat oriMap = Mat::zeros(ori.size(), CV_8UC3);
    Vec3b white(255, 255, 255);
    Vec3b cyan(255, 255, 0);
    Vec3b green(0, 255, 0);
    Vec3b yellow(0, 255, 255);
    for(int i = 0; i < mag.rows*mag.cols; i++)
    {
        float* magPixel = reinterpret_cast<float*>(mag.data + i*sizeof(float));
        if(*magPixel > thresh)
        {
            float* oriPixel = reinterpret_cast<float*>(ori.data + i*sizeof(float));
            Vec3b* mapPixel = reinterpret_cast<Vec3b*>(oriMap.data + i*3*sizeof(char));
            if(*oriPixel <= 150 && *oriPixel >= 130)
                *mapPixel = white;
        }
    }
    
    return oriMap;
}

Mat CannyThreshold(Mat src_gray, int lowThreshold , int highThreshold )
{
    double sigmaX = 1.5;
    double sigmaY = 1.5;
    //int lowThreshold = 100;
    //int highThreshold = 150;
    int kernel_size = 3;
    Mat detected_edges, detected_edges_canny;
    Mat gray, float_gray, blur_img, num, den;
    Mat detected_edges_img;
    
    Mat Sobel_Image;
	src_gray = src_gray > 30;
    GaussianBlur(src_gray, detected_edges, Size(5,5), sigmaX, sigmaY, BORDER_DEFAULT );
    
    Canny( detected_edges, detected_edges_canny, lowThreshold, highThreshold, kernel_size );

	/*vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(detected_edges_canny, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	Mat drawing = Mat::zeros(detected_edges_canny.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255, 255, 255);
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
	}*/

	// Show in a window
	//namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	//imshow("Contours", detected_edges_canny);
	//waitKey(0);
    
  return detected_edges_canny;
 }


Mat CannyThreshold_MemoryImages(Mat src_gray)
{
    double sigmaX = 1.5;
    double sigmaY = 1.5;
    int lowThreshold = 50;
    int highThreshold = 80;
    int kernel_size = 3;
    Mat detected_edges, detected_edges_canny;
    Mat gray, float_gray, blur_img, num, den;
    Mat detected_edges_img;
    float angle =0;
	src_gray = src_gray > 30;
    GaussianBlur(src_gray, detected_edges_canny, Size(5,5), sigmaX, sigmaY, BORDER_DEFAULT );
    Canny( detected_edges_canny, detected_edges_canny, lowThreshold, highThreshold, kernel_size );
    
    Mat croppedImage;
    //printf("src_gray size %d, %d\n", src_gray.rows, src_gray.cols);
    croppedImage = ROI_image(detected_edges_canny);
    
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(detected_edges_canny, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Draw contours
	Mat drawing = Mat::zeros(detected_edges_canny.size(), CV_8UC1);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255);
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
	}

	/// Show in a window
	//namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	//imshow("Contours", drawing);
	//waitKey(0);

	return drawing;
}

Mat resize_image(Mat src, uint64_t target, double *scale_value)
{
    int sigma = 0;
    uint64_t src_pixel_count = countNonZero(src);
    double scale = 1;
    double scale_factor = 0.05;
    uint64_t lower_t = 0.98*target;
    uint64_t upper_t = 1.02*target;
    int iteration_limit=20;
    int iteration_count = 0;
    Mat resized_image=src;
    double fx;
    double fy;
    //int colsv = src.cols;
    //int rowsv = src.rows;
    Size dsize(0,0);
    //int max_iter = 20;
    //int count = 0;

    // Stop the while loop if the number of iterations go beyond 20, otherwise 
    // the pixel count will keep switching between max and min values.
    // The maximum number of times while loop will execute is given by iteration_limit
    // and the counter that keeps track of it is given by imteration_count
    while(src_pixel_count < lower_t || src_pixel_count > upper_t){
	if(src_pixel_count < lower_t){
        	sigma = 1;
    	}else if(src_pixel_count > upper_t){
        	sigma = -1;
    	}
        scale = scale + (sigma)*scale_factor;
        resize(src, resized_image, dsize, scale, scale, INTER_LINEAR );
        src_pixel_count = countNonZero(resized_image);
        iteration_count++;
        if(iteration_count > iteration_limit){
		break;
	}
    }
    *scale_value=scale;
    //printf("src_pixel_count: %llu scale: %g\n", src_pixel_count,scale);
    return resized_image;
}

int fullSizeCount(Mat src)
{
    // Scale up the image to maximum size and see which image has minimum number of pixels.
    Mat resized_image;
    int pixel_count=0;
    double fx=2;
    double fy=2;
    Size dsize(0,0);
    resize(src, resized_image, dsize, fx, fy, INTER_LINEAR );
    pixel_count = countNonZero(resized_image);
    //printf("pixel_count: %d \n", pixel_count);
    //imshow("Full Size image", resized_image*255);
    //cvWaitKey(0);
    return pixel_count;
}

Mat padImageMatrix(Mat inMatrix, int maxRows, int maxCols){
    Mat paddedImage;
    
    int top, bottom, left, right;
    /// Initialize arguments for the filter
    
    //printf("The dimensions of inMatrix (in padImageMatrix fn.) are: %d, %d maxRows and maxCols: %d, %d\n", inMatrix.rows, inMatrix.cols, maxRows, maxCols);
    int mR = (int)(maxRows-inMatrix.rows);
    int mC = (int)(maxCols-inMatrix.cols);
    if(mR%2 == 0){
    	top = mR/2; 
	bottom = mR/2;
    }else{
        top = mR/2;
        bottom = mR/2+1;
    }


    if(mC%2 == 0){
       left = mC/2;
       right = mC/2;
    }else{
        left = mC/2;
        right = mC/2+1;
    }
    copyMakeBorder( inMatrix, paddedImage, top, bottom, left, right, BORDER_CONSTANT, Scalar(0,0,0) );
    
    return paddedImage.clone();
}

Mat foregroundBackgroundImageChange(Mat edgeDetectedImage)
{
    Mat ones_mat = Mat::ones(edgeDetectedImage.rows, edgeDetectedImage.cols, edgeDetectedImage.type());
    Mat changed_image(edgeDetectedImage.rows, edgeDetectedImage.cols, edgeDetectedImage.type());
    changed_image = ones_mat-edgeDetectedImage;
    return changed_image;
}

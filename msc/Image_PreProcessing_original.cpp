#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "Image_PreProcessing.h"
#include <algorithm>
#include <vector>
using namespace cv;
using namespace std;
/// Global variables


/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
Mat CannyThreshold(Mat src_gray, Mat src, Mat dst)
{
    double sigmaX = 1;
    double sigmaY = 1;
    int lowThreshold = 50;
    int highThreshold = 80;
    int kernel_size = 3;
    double sigma = 0.8;
    Mat detected_edges, detected_edges_canny;
    Mat gray, float_gray, blur_img, num, den;
    Mat detected_edges_img;
  /// Reduce noise with a kernel 3x3
  //blur( src_gray, detected_edges, Size(5,5) );
   GaussianBlur(src_gray, detected_edges_canny, Size(5,5), sigmaX, sigmaY, BORDER_DEFAULT );
  //imshow("Detected_Edges_1", detected_edges);
  /// Canny detector
   Canny( detected_edges_canny, detected_edges_canny, lowThreshold, highThreshold, kernel_size );
  //imshow("Detected_Edges_2", detected_edges);
  /// Using Canny's output as a mask, we display our result
  //dst = Scalar::all(0);

  // detected_edges are copied to src.
  //src.copyTo( dst, detected_edges);
    
    
    
 ////////------ Local Contrast Normalization ------------////////////
    
    // convert to floating-point image
    src_gray.convertTo(float_gray, CV_32F, 1.0/255.0);
    
    // numerator = img - gauss_blur(img)
    cv::GaussianBlur(float_gray, blur_img, Size(0,0), 2, 2);
    num = float_gray - blur_img;
    
    // denominator = sqrt(gauss_blur(img^2))
    cv::GaussianBlur(num.mul(num), blur_img, Size(0,0), 20, 20);
    cv::pow(blur_img, 0.5, den);
    
    // output = numerator / denominator
    gray = num / den;
    
    // normalize output into [0,1]
    // normalize(gray, gray, 0.0, 1.0, NORM_MINMAX, -1);
    normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);
    //blur( gray, detected_edges_img, Size(5,5) );
    GaussianBlur(gray, detected_edges, Size(3,3), sigmaX, sigmaY, BORDER_DEFAULT );
    
    
///// -------------------- Adaptive Threshold Technique ------------------------------- ///////////
    /*
    double maxValue = 255.0;
    double C = 0.3;
    adaptiveThreshold(src_gray, detected_edges, maxValue, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, C);
    */
    
/////// ------ Automatically setting the thresholds for canny edge ----------------------/////////
    //GaussianBlur(detected_edges, detected_edges, Size(3,3), sigmaX, sigmaY, BORDER_DEFAULT );
    
    Canny( detected_edges, detected_edges_img, lowThreshold, highThreshold, kernel_size );
    
    
///// ------- Trying to test erosion and dilation ------------------------------------ ///////////
    
    int dilation_size = 6;
    Mat dilation_dst;
    Mat element = getStructuringElement( MORPH_CROSS,
                                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        Point( dilation_size, dilation_size ) );
    dilate( detected_edges_img, dilation_dst, element );
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( detected_edges_img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    Mat drawing = Mat::zeros( detected_edges_img.size(), CV_8UC1 );
    
    for( int i = 0; i< contours.size(); i++ )
    {
        // Calculate contour area
        double area = cv::contourArea(contours[i]);
        
        // Remove small objects by drawing the contour with black color
        if (area > 10){
            drawContours( drawing, contours, i, Scalar(255,255,255), -1);
        }
    }
    /*
    int erosion_size = 3;
    element = getStructuringElement( MORPH_CROSS,
                                    Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                    Point( erosion_size, erosion_size ) );
    /// Apply the erosion operation
    erode( drawing, drawing, element );
    */
    Mat drawing_canny;
    Canny( drawing, drawing_canny, lowThreshold, highThreshold, kernel_size );
    // Display
    
    Mat contoured_plot = drawing_canny.clone();
    
    //floodFill(drawing_canny, Point(0,0), Scalar(0,0,0));
    namedWindow("demo", CV_WINDOW_AUTOSIZE );
    contoured_plot.convertTo(contoured_plot, CV_8UC1);
    imshow("demo", drawing_canny);
    //getchar();
    cvWaitKey(0);
    /*
    namedWindow("demo", CV_WINDOW_AUTOSIZE );
    contoured_plot.convertTo(contoured_plot, CV_8UC1);
    imshow("demo", detected_edges_img);
    */
    
    dst = Scalar::all(0);
    
    // detected_edges are copied to src.
    src_gray.copyTo( dst, contoured_plot);
    
  ////////----------------- End of Code --------------------//////////////
    
    
    
  return detected_edges_canny;
 }


Mat CannyThreshold_MemoryImages(Mat src_gray, Mat src, Mat dst)
{
    double sigmaX = 1;
    double sigmaY = 1;
    
    int edgeThresh = 1;
    int lowThreshold = 50;
    int highThreshold = 80;
    int ratio = 3;
    int kernel_size = 3;
    double sigma = 0.8;
    Mat detected_edges, detected_edges_canny;
    Mat gray, float_gray, blur_img, num, den;
    Mat detected_edges_img;
    
    
    GaussianBlur(src_gray, detected_edges_canny, Size(3,3), sigmaX, sigmaY, BORDER_DEFAULT );
    
    Canny( detected_edges_canny, detected_edges_canny, lowThreshold, highThreshold, kernel_size );
    
    
    ////////------ Local Contrast Normalization ------------////////////
    
    // convert to floating-point image
    src_gray.convertTo(float_gray, CV_32F, 1.0/255.0);
    
    // numerator = img - gauss_blur(img)
    cv::GaussianBlur(float_gray, blur_img, Size(0,0), 2, 2);
    num = float_gray - blur_img;
    
    // denominator = sqrt(gauss_blur(img^2))
    cv::GaussianBlur(num.mul(num), blur_img, Size(0,0), 20, 20);
    cv::pow(blur_img, 0.5, den);
    
    // output = numerator / denominator
    gray = num / den;
    
    // normalize output into [0,1]
    // normalize(gray, gray, 0.0, 1.0, NORM_MINMAX, -1);
    normalize(gray, gray, 0, 255, NORM_MINMAX, CV_8UC1);
    //blur( gray, detected_edges_img, Size(5,5) );
    GaussianBlur(gray, detected_edges, Size(3,3), sigmaX, sigmaY, BORDER_DEFAULT );
    
    
    Canny( detected_edges, detected_edges_img, lowThreshold, highThreshold, kernel_size );
    
    
    ///// ------- Trying to test erosion and dilation ------------------------------------ ///////////
    
    int dilation_size = 6;
    Mat dilation_dst;
    Mat element = getStructuringElement( MORPH_CROSS,
                                        Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        Point( dilation_size, dilation_size ) );
    dilate( detected_edges_canny, dilation_dst, element );
    
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( detected_edges_img, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    Mat drawing = Mat::zeros( detected_edges_img.size(), CV_8UC1 );
    
    for( int i = 0; i< contours.size(); i++ )
    {
        // Calculate contour area
        double area = cv::contourArea(contours[i]);
        
        // Remove small objects by drawing the contour with black color
        if (area > 10){
            drawContours( drawing, contours, i, Scalar(255,255,255), 1 );
        }
    }
    
    
    
    /*
    Mat DC = dilation_dst.clone();
    Mat flood;
    floodFill(DC, Point(0,0), Scalar(255, 255,255));
    bitwise_not(DC,DC);
    flood = (DC|dilation_dst);
    */
    /*
    int erosion_size = 6;
    element = getStructuringElement( MORPH_CROSS,
                                        Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                        Point( erosion_size, erosion_size ) );
    /// Apply the erosion operation
    erode( flood, flood, element );
    */
    Mat final_edge_img;
    Mat drawing_canny;
    Canny( drawing, drawing_canny, lowThreshold, highThreshold, kernel_size );
    
    final_edge_img = drawing_canny.clone();
    
    final_edge_img.convertTo(final_edge_img, CV_8UC1);
    
    namedWindow("demo", CV_WINDOW_AUTOSIZE );
    
    imshow("demo", drawing_canny);
    //getchar();
    cvWaitKey(0);
    dst = Scalar::all(0);
    
    // detected_edges are copied to src.
    //src_gray.copyTo( dst, final_edge_img);
    
    ////////----------------- End of Code --------------------//////////////
    
    
    
    return final_edge_img;
}

double medianMat(Mat Input, int nVals){
    
    // COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
    float range[] = { 0, nVals };
    const float* histRange = { range };
    bool uniform = true; bool accumulate = false;
    cv::Mat hist;
    calcHist(&Input, 1, 0, cv::Mat(), hist, 1, &nVals, &histRange, uniform, accumulate);
    
    // COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
    cv::Mat cdf;
    hist.copyTo(cdf);
    for (int i = 1; i <= nVals-1; i++){
        cdf.at<float>(i) += cdf.at<float>(i - 1);
    }
    cdf /= Input.total();
    
    // COMPUTE MEDIAN
    double medianVal=0.0;
    for (int i = 0; i <= nVals-1; i++){
        if (cdf.at<float>(i) >= 0.5) { medianVal = i;  break; }
    }
    return medianVal/nVals;
}

int max (double b) {
    int a = 0;
    return ((int)b<a)?a:(int)b;     // or: return !comp(b,a)?a:b; for version (2)
}

int min (double b) {
    int a = 255;
    return ((int)b<a)?(int)b:a;     // or: return !comp(b,a)?a:b; for version (2)
}

Mat padImageMatrix(Mat inMatrix){
    Mat paddedImage;
    
    int top, bottom, left, right;
    /// Initialize arguments for the filter
    
    //printf("The dimensions of inMatrix (in padImageMatrix fn.) are: %d, %d\n", inMatrix.rows, inMatrix.cols);
    top = (int) (0); bottom = (int) (400-inMatrix.rows);
    left = (int) (0); right = (int) (600-inMatrix.cols);
    copyMakeBorder( inMatrix, paddedImage, top, bottom, left, right, BORDER_CONSTANT, Scalar(0,0,0) );
    
    return paddedImage.clone();
}

/// -------- Canny threshold detection using Otsu's threshold --------------------- ////////
//Mat _img;
//double otsu_thresh_val = cv::threshold(detected_edges_img, _img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

//double high_thresh_val  = otsu_thresh_val, lower_thresh_val = otsu_thresh_val * 0.5;
//cv::Canny( detected_edges_img, detected_edges_img, lower_thresh_val, high_thresh_val );

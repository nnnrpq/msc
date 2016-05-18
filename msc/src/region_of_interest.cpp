//
//  main.cpp
//  
//
//  Created by Rohit Shukla on 9/26/15.
//
//

#include <stdio.h>
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include "region_of_interest.h"

using namespace cv;
using namespace std;

Mat ROI_image(Mat src)
{
    // Store the set of points in the image before assembling the bounding box
    std::vector<cv::Point> points;
    cv::Mat_<uchar>::iterator it = src.begin<uchar>();
    cv::Mat_<uchar>::iterator end = src.end<uchar>();
    for (; it != end; ++it)
    {
        if (*it) points.push_back(it.pos());
    }
    
    // Compute minimal bounding box
    cv::RotatedRect box = cv::minAreaRect(cv::Mat(points));
    
    // Set Region of Interest to the area defined by the box
    cv::Rect roi;
    //cout<<box.center.x<<"   "<<box.size.width<<"   "<<box.center.y<<"   "<<box.size.height<<endl;
    roi.x = box.center.x - (box.size.width / 2);
    roi.y = box.center.y - (box.size.height / 2);
    roi.width = box.size.width;
    roi.height = box.size.height;
    if(roi.x < 0)
        roi.x=(src.cols / 2);
    if(roi.y < 0)
        roi.y=(src.rows / 2);
    if(roi.width >= src.cols)
        roi.width=src.cols/2;
    if(roi.height >= src.rows)
        roi.height=src.rows/2;
    //cout<<roi.x<<"   "<<roi.width<<"   "<<roi.y<<"   "<<roi.height<<endl;
    
    // Crop the original image to the defined ROI
    cv::Mat crop = src(roi);
    return crop;
}


Mat Resize_image(Mat src)
{
    Size size(600,400);
    Mat resized;
    double scale = 1;
    resize(src, resized, size, scale, scale, INTER_LINEAR );
    
    return resized;
}

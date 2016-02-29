#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "./Image_PreProcessing.h"
#include <vector>
#include "MSC.h"
//#include <Windows.h>
#include "dirent.h"
#include "Generate_Image.h"
#include "Learn_Image.h"

using namespace std;
using namespace cv;

Mat src, src_gray;
Mat dst;
double mul = 0.9;		/* the rate of updating the test image*/
double th = 0.05;		/* threshold for whether a transformation from msc is qualified*/


struct filename_struct {
	char filename[200];
};



int main() {
	int buf = 0;
	Mat Memory_Images;
	Mat Memory_Images_background;
	Size img_size;
	vector<Mat> cropped_memory_images;
	vector<Mat> cropped_memory_images_background;
	Mat Fwd_Image;
	Mat Bwd_Image;
	double Thresh_VAL = 100;
	double MAX_VAL = 1;
	uint64_t min_pixel_count = 240000;
	uint64_t max_pixel_count = 0;
	//char c[] = "Caltech-101/";
	//char* input_image = ".\\101_ObjectCategories\\car_side\\image_0005.jpg";


	DIR *dir;
	struct dirent *ent;
	vector< filename_struct > filename_vector;

	Mat test = imread("template.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	/*step 1, generate random image*/
	double theta, xtran, ytran, scale;
	imshow("template", test);
	Mat src = Rand_Transform(test, theta, xtran, ytran, scale);		/*src is the test image*/
	imshow("random image", src);
	waitKey();

	bool flag = 1;	/* flag for whether to continue the transformation*/

	do {

		/// Create a matrix of the same type and size as src (for dst)

		src_gray = src.clone();
		dst.create(src_gray.size(), src_gray.type());
		/// Perform cannyThreshold operation (This function is present as separate C++ file)
		Mat Edge_Detected_Image_GrayScale = CannyThreshold(src_gray, src, dst).clone();
		Mat Edge_Detected_Image_unpadded(src_gray.rows, src_gray.cols, CV_32FC1);

		// The image pixel values should be in range between 0 and 1. Therefore, performing the
		// threshold function on the edge detected image.
		threshold(Edge_Detected_Image_GrayScale, Edge_Detected_Image_unpadded, Thresh_VAL, MAX_VAL, THRESH_BINARY);
		printf("Edge detect image done Size is (%d, %d)\n", Edge_Detected_Image_unpadded.rows, Edge_Detected_Image_unpadded.cols);

		/// MSC implementation
			// This indicates that there are images already present in the memory.

		buf = 1;
		// Step 1 :- Parse through all of the memory images and get the 
		// non pixel count. At the same time store the ROI images in the 
		// Mat file. No need to pad images in this step.  


		Mat mem_img_gray = test;					/* use the template image as memory image*/
		Mat mem_img_canny = CannyThreshold_MemoryImages(mem_img_gray);


		// Since the image pixel values should be between 0 and 1, as a result,
		// reflecting the values of g, thus, performing the edge detection function.
		Mat mem_img_edge(mem_img_canny.rows, mem_img_canny.cols, CV_32FC1);
		threshold(mem_img_canny, mem_img_edge, Thresh_VAL, MAX_VAL, THRESH_BINARY);


		// Store all of the edge detected images in the vector.
		// Reshape the matrices to just a single row and push these
		// values to the Memory_Images matrix variable.
		cropped_memory_images.push_back(mem_img_edge);
		// Store the number of rows present in the original image.


	// Step 2:- Scroll through all of the images and perform the pixel normalization. There are couple of ways to approach this
	// problem. (Approach 1) Bring the pixels of all the images close to the one that has the maximum value. A while loop will
	// keep on changing the scale factor until the pixel count of input image becomes equal to the maximum pixel count.
		int maxRows = Edge_Detected_Image_unpadded.rows;
		int maxCols = Edge_Detected_Image_unpadded.cols;


		for (int i = 0; i < buf; i++) {

			if (maxRows < cropped_memory_images[i].rows) {
				maxRows = cropped_memory_images[i].rows;
			}
			if (maxCols < cropped_memory_images[i].cols) {
				maxCols = cropped_memory_images[i].cols;
			}

		}

		//Resize the input image as well the memory images
		Mat Edge_Detected_Image(maxRows + 1, maxCols + 1, CV_32FC1);
		Edge_Detected_Image = padImageMatrix(Edge_Detected_Image_unpadded, maxRows + 1, maxCols + 1);

		for (int i = 0; i < buf; i++) {
			// Before padding the image, keep the edge image values at 1 and background as 0.
			Mat paddedImage = padImageMatrix(cropped_memory_images[i], maxRows + 1, maxCols + 1);
			Memory_Images.push_back(paddedImage.reshape(0, 1));

			printf("The values of padded image %d, %d \n", paddedImage.rows, paddedImage.cols);
		}
		img_size = Size(maxCols + 1, maxRows + 1);

		// The actual MSC will go over here.
		TransformationSet finalTrans;
		int ret = SL_MSC(Edge_Detected_Image, Memory_Images, img_size, &Fwd_Image, &Bwd_Image, finalTrans);
		printf("The return value of SL_MSC is %d\n", ret);

		// Get the returned address Images.
		imshow("Forward Path", Fwd_Image * 255);
		imshow("Backward Path", Bwd_Image * 255);

		/// Wait until user exit program by pressing a key
		waitKey(0);

		/* check the result of msc and update image*/
		flag = finalTrans.nonIdenticalCount != 0;
		printf("the target xtran is %f, msc result is %f\n", xtran, finalTrans.xTranslate);
		printf("the target ytran is %f, msc result is %f\n", ytran, finalTrans.yTranslate);
		printf("the target theta is %f, msc result is %f\n", theta, finalTrans.theta);
		printf("the target scale is %f, msc result is %f\n", scale, finalTrans.scale);

		if (abs(xtran - finalTrans.xTranslate) / img_size.width < th && abs(ytran - finalTrans.yTranslate) / img_size.height < th&&
			abs(scale - finalTrans.scale) < th && abs(theta - finalTrans.theta) / 360 < th) {
			xtran = xtran*mul;
			ytran = xtran*mul;
			scale = scale*mul;
			theta = theta*mul;
			src = Rand_Transform(test, theta, xtran, ytran, scale);
		}
		else {
			printf("MSC is wrong\n");
			flag = 1;
		}
	} while (flag);
	printf("MSC is done\n");

	return 0;

}
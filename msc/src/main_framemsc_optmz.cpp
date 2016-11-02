/*	This is the up-to-date main testbench function for testing.
	Frame by frame relationship is considered.
		- Zhangyuan Wang 06/12/2016
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "./Image_PreProcessing.h"
#include <vector>
#include <ctime>
#include "MSC.h"
//#include <Windows.h>
#include "dirent.h"
#include "Generate_Image.h"
#include "Learn_Image.h"


#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

//#ifdef _DEBUG
//#ifndef DBG_NEW
//#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
//#define new DBG_NEW
//#endif
//#endif  // _DEBUG


using namespace std;
using namespace cv;

Mat src, src_gray;
Mat dst;
double mul = 0.95;		/* the rate of updating the test image (it will get closer to the identity transformation)*/
double th = 0.15;		/* manully selected threshold for whether a transformation from msc is qualified*/					
int framectl = 1;		/* frame control, whether to use frame by frame continuity*/
bool dispresult = 0;	/* whether to display result from the MSC (shows an overlap version of the target and src after MSC) */

struct filename_struct {
	char filename[200];
};



int main() {
	int buf = 0;
	Mat Memory_Images;
	Mat Memory_Images_background;
	Size img_size;
	Mat cropped_memory_images;
	//vector<Mat> cropped_memory_images_background;
	Mat Fwd_Image;
	Mat Bwd_Image;
	double Thresh_VAL = 100;
	double MAX_VAL = 1;

	vector< filename_struct > filename_vector;


/*	Select test image.
	img_1.png is the duck image.
	apriltag0.png is one of the april tag files. To use it, we need another preprocessing technique
	templatePY.jpg is the manually generated image of letter P and Y*/

	//Mat test = imread("apriltag0.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat test = imread("img_1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat test = imread("templatePY.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat src = imread("js/logo20.png", CV_LOAD_IMAGE_GRAYSCALE);


/*	This part is supposed to do dialation to the (PY) image, 
	in order to test how the different width of edges can effect the result.
	It seems that the difference is not obvious*/

	//Mat test1;
	//dilate(test, test1, Mat::ones(Size(15, 15), CV_8U));
	//imshow("before", test);
	////test = test1;
	//imshow("after dilate", test1);
	//waitKey();

/* Enlarge the image by 1.5, and pad with zero,
	to avoid the object from moving out of the frame in the random transformation step*/
	//test = padImageMatrix(test, round(test.rows*1.5), round(test.cols*1.5));
	
	/*step 1, generate random image*/
	//double theta, xtran, ytran, scale;
	//imshow("template", test);
	//src = Rand_Transform(src, theta, xtran, ytran, scale,1);		/*src is the random transformed image*/
	//src = imread("b1.png", CV_LOAD_IMAGE_GRAYSCALE);

	//src.convertTo(src, CV_32FC1, 1.0 / 255);


	bool flag = 1;	/* flag for whether to continue the transformation*/

	TransformationSet finalTrans;		/* The object to store previously calculated result*/
	finalTrans.nonIdenticalCount = -1;	/* Initialization*/

	int count = 0;						/* count how many iterations has been done*/
	int maxiter = 30;					/* proceed until maxiter*/
	UINT t1, t2;						/* timer*/

	Mat Edge_Detected_Image;
	src_gray = src.clone();

	//Mat mem_img_gray = src;					/* use the original template image as memory image*/
	Mat mem_img_gray = imread("img_1.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat mem_img_canny = CannyThreshold_MemoryImages(mem_img_gray);
	Mat mem_img_edge(mem_img_canny.rows, mem_img_canny.cols, CV_32FC1);
	threshold(mem_img_canny, mem_img_edge, Thresh_VAL, MAX_VAL, THRESH_BINARY);
	Mat Edge_Detected_Image_GrayScale;
	Mat Edge_Detected_Image_unpadded(src_gray.rows, src_gray.cols, CV_32FC1);

	int ret;
	//do {
		if (count == 1)
			t1 = clock();


		/* Preprocessing*/
		Edge_Detected_Image_GrayScale = CannyThreshold(src_gray).clone();
		threshold(Edge_Detected_Image_GrayScale, Edge_Detected_Image_unpadded, Thresh_VAL, MAX_VAL, THRESH_BINARY);
		Edge_Detected_Image_unpadded.convertTo(Edge_Detected_Image_unpadded, CV_32FC1);

		/*int maxRows = Edge_Detected_Image_unpadded.rows;
		int maxCols = Edge_Detected_Image_unpadded.cols;*/

		int width = src.cols;
		int height = src.rows;
		int maxRows = Edge_Detected_Image_unpadded.rows;
		int maxCols = Edge_Detected_Image_unpadded.cols;
		/*if (maxRows < cropped_memory_images.rows) {
			maxRows = cropped_memory_images.rows;
		}
		if (maxCols < cropped_memory_images.cols) {
			maxCols = cropped_memory_images.cols;
		}*/

		imshow("Input", Edge_Detected_Image_unpadded);
		waitKey(0);
		resize(Edge_Detected_Image_unpadded, Edge_Detected_Image_unpadded, Size(maxCols, maxRows));

		Memory_Images = padImageMatrix(mem_img_edge, maxRows, maxCols);
		img_size = Memory_Images.size();

		imshow("Memory", Memory_Images);
		waitKey(0);
		

		/* control whether or not to do the frame by frame optimization*/
		if (!framectl)
			finalTrans.nonIdenticalCount = -1;

		/* do MSC*/
		ret = SL_MSC(Edge_Detected_Image_unpadded, Memory_Images, img_size, &Fwd_Image, &Bwd_Image, finalTrans);
		
		//printf("The return value of SL_MSC is %d\n", ret);


		/* check the result of msc and update image*/
		flag = finalTrans.nonIdenticalCount != 0;
		printf("x-tran msc result is %f\n", finalTrans.xTranslate);
		printf("y-tran msc result is %f\n", finalTrans.yTranslate);
		printf("rot msc result is %f\n", finalTrans.theta);
		printf("scale msc result is %f\n", finalTrans.scale);

		// Get the returned address Images.
		//imshow("Forward Path", Fwd_Image * 255);
		if (dispresult) {
			imshow("Backward Path", (Edge_Detected_Image_unpadded + Bwd_Image) * 255);
			waitKey(0);
		}


		/*if (abs(xtran - finalTrans.xTranslate) / img_size.width < th && abs(ytran - finalTrans.yTranslate) / img_size.height < th&&
			abs(scale - finalTrans.scale) < th && abs(theta - finalTrans.theta) / 180 < th) {
			printf("MSC is right\n");
		}
		else {
			printf("MSC is wrong\n");
			flag = 0;
		}*/

		/* update image*/
		/*xtran = xtran*mul;
		ytran = ytran*mul;
		scale = 1 - (1 - scale)*mul;
		theta = theta*mul;
		src_gray = Rand_Transform(test, theta, xtran, ytran, scale, 2);

		count++;
	} while (count<maxiter);*/
	t2 = clock();

//	_CrtDumpMemoryLeaks();
	printf("MSC is done\n");
	printf("time for %d iterations is %d\n", count, t2 - t1);
	printf("Correlation Value: %d", ret);

	getchar();

	imshow("Backward Path", (Edge_Detected_Image_unpadded + Bwd_Image) * 255);
	waitKey(0);
	return 0;

}
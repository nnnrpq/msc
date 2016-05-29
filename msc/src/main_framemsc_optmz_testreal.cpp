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
double mul = 0.95;		/* the rate of updating the test image*/
double th = 0.15;		/* threshold for whether a transformation from msc is qualified*/					
int framectl = 1;		/* frame control*/
bool dispresult = 1;

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

	//char c[] = "Caltech-101/";
	//char* input_image = ".\\101_ObjectCategories\\car_side\\image_0005.jpg";

	DIR *dir;
	struct dirent *ent;
	vector< filename_struct > filename_vector;

	Mat C = (Mat_<double>(2, 3) << 0, 0, 1,2,3,4);


	//Mat test = imread("apriltag0.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat test = imread("img_1.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat test = imread("templatePY.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	//Mat test1;
	//dilate(test, test1, Mat::ones(Size(15, 15), CV_8U));
	//imshow("before", test);
	////test = test1;
	//imshow("after dilate", test1);
	//waitKey();

	//test = padImageMatrix(test, round(test.rows*1.5), round(test.cols*1.5));
	/*step 1, generate random image*/
	double theta, xtran, ytran, scale;
	//imshow("template", test);
	Mat src = Rand_Transform(test, theta, xtran, ytran, scale,1);		/*src is the test image*/
	src = imread("b1.png", CV_LOAD_IMAGE_GRAYSCALE);
	int width = src.cols;
	int height = src.rows;
	//src.convertTo(src, CV_32FC1, 1.0 / 255);


	bool flag = 1;	/* flag for whether to continue the transformation*/
	buf = 1;
	TransformationSet finalTrans;
	finalTrans.nonIdenticalCount = -1;
	int count = 0;
	int maxiter = 30;
	UINT t1, t2;
	Mat Edge_Detected_Image;
	src_gray = src.clone();
	/// Perform cannyThreshold operation (This function is present as separate C++ file)
	//imshow("src", src);
	//waitKey();

	Mat mem_img_gray = test;					/* use the template image as memory image*/
	Mat mem_img_canny = CannyThreshold_MemoryImages(mem_img_gray);
	// Since the image pixel values should be between 0 and 1, as a result,
	// reflecting the values of g, thus, performing the edge detection function.
	Mat mem_img_edge;//(mem_img_canny.rows, mem_img_canny.cols, CV_32FC1);
	threshold(mem_img_canny, mem_img_edge, Thresh_VAL, MAX_VAL, THRESH_BINARY);
	mem_img_edge.convertTo(mem_img_edge, CV_32FC1);
	img_size = mem_img_edge.size();
	Mat Edge_Detected_Image_GrayScale;
	Mat Edge_Detected_Image_unpadded(src_gray.rows, src_gray.cols, CV_32FC1);
	do {
		Edge_Detected_Image_GrayScale = CannyThreshold(src_gray).clone();
		threshold(Edge_Detected_Image_GrayScale, Edge_Detected_Image_unpadded, Thresh_VAL, MAX_VAL, THRESH_BINARY);
		Edge_Detected_Image_unpadded.convertTo(Edge_Detected_Image_unpadded, CV_32FC1);

		if (count == 1)
			t1 = clock();
		//imshow("random image", src);
		//waitKey();

		/// Create a matrix of the same type and size as src (for dst)


		int maxRows = test.rows;
		int maxCols = test.cols*width / height;

		Mat Memory_Images = padImageMatrix(mem_img_edge, maxRows, maxCols);

		resize(Edge_Detected_Image_unpadded, Edge_Detected_Image_unpadded, Size(maxCols, maxRows));

		Size img_size = Memory_Images.size();

		if (!framectl)
			finalTrans.nonIdenticalCount = -1;
		//imshow("src", Edge_Detected_Image_unpadded);
		//waitKey();

		int ret = SL_MSC(Edge_Detected_Image_unpadded, Memory_Images, img_size, &Fwd_Image, &Bwd_Image, finalTrans);

		


		
		//printf("The return value of SL_MSC is %d\n", ret);


		/* check the result of msc and update image*/
		flag = finalTrans.nonIdenticalCount != 0;
		printf("the target xtran is %f, msc result is %f\n", xtran, finalTrans.xTranslate);
		printf("the target ytran is %f, msc result is %f\n", ytran, finalTrans.yTranslate);
		printf("the target theta is %f, msc result is %f\n", theta, finalTrans.theta);
		printf("the target scale is %f, msc result is %f\n", scale, finalTrans.scale);

		// Get the returned address Images.
		//imshow("Forward Path", Fwd_Image * 255);
		if (dispresult) {
			imwrite("result.png", 255*(Edge_Detected_Image_unpadded + Bwd_Image));
			imshow("Backward Path", (Edge_Detected_Image_unpadded + Bwd_Image));
			waitKey(0);
		}


		if (abs(xtran - finalTrans.xTranslate) / img_size.width < th && abs(ytran - finalTrans.yTranslate) / img_size.height < th&&
			abs(scale - finalTrans.scale) < th && abs(theta - finalTrans.theta) / 180 < th) {
			printf("MSC is right\n");
		}
		else {
			printf("MSC is wrong\n");
			flag = 0;
		}
		//waitKey();
		xtran = xtran*mul;
		ytran = ytran*mul;
		scale = 1 - (1 - scale)*mul;
		theta = theta*mul;
		src_gray = Rand_Transform(test, theta, xtran, ytran, scale, 2);
		//buf = 0;
		count++;
	} while (count<maxiter);
	t2 = clock();

//	_CrtDumpMemoryLeaks();
	printf("MSC is done\n");
	printf("time for %d iterations is %d\n", count, t2 - t1);

	getchar();

	imshow("Backward Path", (Edge_Detected_Image_unpadded + Bwd_Image) * 255);
	waitKey(0);
	return 0;

}
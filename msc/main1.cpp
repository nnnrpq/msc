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
#include <Windows.h>
#include "dirent.h"

using namespace std;
using namespace cv;

Mat src, src_gray;
Mat dst;

struct filename_struct {
	char filename[200];
};

int main() {
	int buf = 0;
	Mat Memory_Images;
	Mat Memory_Images_background;
	vector<int> row_size;
	vector<Mat> cropped_memory_images;
	vector<Mat> cropped_memory_images_background;
	Mat Fwd_Image;
	Mat Bwd_Image;
	double Thresh_VAL = 100;
	double MAX_VAL = 1;
	uint64_t min_pixel_count = 240000;
	uint64_t max_pixel_count = 0;
	char c[] = "Caltech-101/";
	char* input_image = ".\\101_ObjectCategories\\car_side\\image_0005.jpg";
	int maxRows = 0;
	int maxCols = 0;

	DIR *dir;
	struct dirent *ent;
	vector< filename_struct > filename_vector;

	/// Load an image
	src = imread(input_image, CV_LOAD_IMAGE_GRAYSCALE);
	//src_gray = src.clone();
	if (!src.data)
	{
		printf("Specified file not found \n");
		return -1;
	}

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
	


	buf = 0;
	if ((dir = opendir("Memory_Images/")) != NULL) {
		/* print all the files and directories within directory */
		while ((ent = readdir(dir)) != NULL) {
			buf++;
			printf("%s %d\n", ent->d_name, buf);
			if (buf >= 4) {
				filename_struct s;
				strcpy(s.filename, "Memory_Images/");
				strcat(s.filename, ent->d_name);
				filename_vector.push_back(s);
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		perror("");
		return EXIT_FAILURE;
	}
	buf = buf - 3;
	
	/// MSC implementation
	if (buf == 0) {
		//// If the value of buf is zero this means that
		//// MSC does not have any image stored in its memory.
		//Mat stored_image;
		//buf = 1;
		//if (argc == 3) {
		//	if (strcmp(argv[2], "Caltech-101") == 0) {
		//		char segmented_image_folder[] = "./Caltech-101/Segmented_Image/";
		//		char* memory_image_stored = strcat(segmented_image_folder, argv[1]);
		//		printf("The path for memory_image_stored is: %s\n", memory_image_stored);
		//		stored_image = imread(memory_image_stored);
		//	}
		//}
		//else {
		//	stored_image = imread(argv[1]);
		//}
		//imwrite("./Memory_Images/Img_1.jpg", stored_image);
		//FILE *fp = fopen("Memory_Image_Count.txt", "w");
		//fprintf(fp, "%d", buf);
		//fclose(fp);
	}
	else {
		// This indicates that there are images already present in the memory.


		// Step 1 :- Parse through all of the memory images and get the 
		// non pixel count. At the same time store the ROI images in the 
		// Mat file. No need to pad images in this step.  
		for (int i = 1; i<buf + 1; i++) {
			// Read all of the images present in the Memory_Images folder

			/*
			char integer_value[8];
			char filename[200];
			int n = sprintf(integer_value, "%d", i);
			strcpy(filename, "Memory_Images/img_");
			strcat(filename, integer_value);
			strcat(filename,".jpg");
			*/


			Mat mem_img_gray = imread(filename_vector[i - 1].filename, CV_LOAD_IMAGE_GRAYSCALE);
			Mat mem_img_canny = CannyThreshold_MemoryImages(mem_img_gray);


			// Since the image pixel values should be between 0 and 1, as a result,
			// reflecting the values of g, thus, performing the edge detection function.
			Mat mem_img_edge(mem_img_canny.rows, mem_img_canny.cols, CV_32FC1);
			threshold(mem_img_canny, mem_img_edge, Thresh_VAL, MAX_VAL, THRESH_BINARY);
			printf("Edge detect of memory image %d done Size is (%d, %d)\n", i, mem_img_edge.rows, mem_img_edge.cols);


			// Store all of the edge detected images in the vector.
			// Reshape the matrices to just a single row and push these
			// values to the Memory_Images matrix variable.
			cropped_memory_images.push_back(mem_img_edge);
			// Store the number of rows present in the original image.
		}


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

		for (int i = 0; i< buf; i++) {
			// Before padding the image, keep the edge image values at 1 and background as 0.
			Mat paddedImage = padImageMatrix(cropped_memory_images[i], maxRows + 1, maxCols + 1);
			Memory_Images.push_back(paddedImage.reshape(0, 1));

			row_size.push_back(paddedImage.rows);

			printf("The values of padded image %d, %d \n", paddedImage.rows, paddedImage.cols);
		}
		// The actual MSC will go over here.
		int ret = SL_MSC(Edge_Detected_Image, Memory_Images, row_size, &Fwd_Image, &Bwd_Image);
		printf("The return value of SL_MSC is %d\n", ret);

		// Get the returned address Images.
		imshow("Forward Path", Fwd_Image * 255);
		imshow("Backward Path", Bwd_Image * 255);

	}
	/// Wait until user exit program by pressing a key
	waitKey(0);
	return 0;

}
//
//bool ListDirectoryContents(const wchar_t *sDir,vector<wchar_t> &file_name)
//{
//	WIN32_FIND_DATA fdFile;
//	HANDLE hFind = NULL;
//
//	wchar_t sPath[2048];
//
//	//Specify a file mask. *.* = We want everything! 
//	wsprintf(sPath, L"%s\\*.*", sDir);
//
//	if ((hFind = FindFirstFile(sPath, &fdFile)) == INVALID_HANDLE_VALUE)
//	{
//		wprintf(L"Path not found: [%s]\n", sDir);
//		return false;
//	}
//
//	do
//	{
//		//Find first file will always return "."
//		//    and ".." as the first two directories. 
//		if (wcscmp(fdFile.cFileName, L".") != 0
//			&& wcscmp(fdFile.cFileName, L"..") != 0)
//		{
//			//Build up our file path using the passed in 
//			//  [sDir] and the file/foldername we just found: 
//			wsprintf(sPath, L"%s\\%s", sDir, fdFile.cFileName);
//			file_name.push_back(sPath);
//			//Is the entity a File or Folder? 
//			if (fdFile.dwFileAttributes &FILE_ATTRIBUTE_DIRECTORY)
//			{
//				wprintf(L"Directory: %s\n", sPath);
//			}
//			else {
//				wprintf(L"File: %s\n", sPath);
//			}
//		}
//	} while (FindNextFile(hFind, &fdFile)); //Find the next file. 
//
//	FindClose(hFind); //Always, Always, clean things up! 
//
//	return true;
//}
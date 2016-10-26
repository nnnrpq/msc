//
//  Learn_Image.cpp
//  
//
//  Created by Rohit Shukla on 7/21/15.
//
//	Modified by Zhangyuan Wang
//	Feb 21, add feature detector, matching and computing
#ifdef WIN32
#pragma warning( push ) 
#pragma warning( disable: 4530 )
namespace std { typedef type_info type_info; }
#endif
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/flann/flann.hpp"
#include "opencv2/video/tracking.hpp"

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "Learn_Image.h"

using namespace cv;
using namespace std;

double Verify_Object(Mat Input_Object, Mat Backward_Path_Object, double dot_product_input_object){
    double dot_product_transformations;
    Backward_Path_Object.convertTo(Backward_Path_Object, CV_32FC1);
    Input_Object.convertTo(Input_Object, CV_32FC1);
    dot_product_transformations = Input_Object.dot(Backward_Path_Object)/ sqrt(Input_Object.dot(Input_Object)* Backward_Path_Object.dot(Backward_Path_Object));
    
	//imshow("input", Input_Object);
	//imshow("result", Backward_Path_Object);
	//waitKey();

    if(dot_product_transformations < 0.2*dot_product_input_object){
        return 0;
    }

    return 1;
}

Mat Learn_New_Transformation(Mat Input_Image, Mat Memory_Images, vector<int> row_size){
    int Memory_Image_Count = Memory_Images.rows;
    Mat tmp_Memory_Image;
    Mat Projective_Transformation;
    for(int i=0; i<Memory_Image_Count; i++){
        tmp_Memory_Image = Memory_Images.reshape(0,row_size[i]);
        Projective_Transformation = Image_Match(Input_Image, tmp_Memory_Image);
    }
    
    return Projective_Transformation;
}

Mat Image_Match(Mat img_object, Mat img_scene){
    Mat perspective_transformation = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    
    if( !img_object.data || !img_scene.data )
    { std::cout<< " --(!) Error reading images " << std::endl; return perspective_transformation; }
    
    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    
    SurfFeatureDetector detector( minHessian );
    
    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    
    detector.detect( img_object, keypoints_object );
    detector.detect( img_scene, keypoints_scene );
    
    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;
    
    Mat descriptors_object, descriptors_scene;
    
    extractor.compute( img_object, keypoints_object, descriptors_object );
    extractor.compute( img_scene, keypoints_scene, descriptors_scene );
    
    //-- Step 3: Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_scene, matches );
    
    double max_dist = 0; double min_dist = 100;
    
    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_object.rows; i++ )
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
    
    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector< DMatch > good_matches;
    
    for( int i = 0; i < descriptors_object.rows; i++ )
    { if( matches[i].distance < 3*min_dist )
    { good_matches.push_back( matches[i]); }
    }
    
    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    
    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    
    Mat H = findHomography( obj, scene, CV_RANSAC );
	//Mat H = estimateRigidTransform(obj, scene, false);
	//Mat H = MyfindAffine(obj, scene, CV_RANSAC,10000);
    
    ////-- Get the corners from the image_1 ( the object to be "detected" )
    //std::vector<Point2f> obj_corners(4);
    //obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
    //obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
    //std::vector<Point2f> scene_corners(4);
    //
    //perspectiveTransform( obj_corners, scene_corners, H);
	


    ////-- Draw lines between the corners (the mapped object in the scene - image_2 )
    //line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
    //line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    //line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    //line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    //
    //-- Show detected matches
    imshow( "Good Matches & Object detection", img_matches );
	waitKey();

	//-- Show the transformed image
	Mat transformed;
	cout << H << endl;
	//warpAffine(img_object, transformed, H, Size(img_object.rows, img_object.cols));
	warpPerspective(img_object, transformed, H, Size(img_object.rows, img_object.cols));

	imshow("transformed image", transformed);
	waitKey();

    return perspective_transformation;
}

//! Find 6 DOF affine transformation from point match between obj and scene
// Use CV_RANSAC or 0 (use all keypoints) to specify the type of algorithm to use
Mat MyfindAffine(vector<Point2f> obj, vector<Point2f> scene, int type,int maxtrail) {
	if (type == 0) {
		Mat H = getAffineTransform(obj, scene);
		return H;
	}
	else if (type != CV_RANSAC) {
		printf("no such find affine method:input 0 or CV_RANSAC\n");
		return Mat();
	}
	else {
		/*ransac method*/

		/*normalise and augment the coodinate first*/
		Mat Tobj, Tscene;
		vector<Point2f> newobj, newscene;
		//mynormalize(obj, newobj, Tobj);
		//mynormalize(scene, newscene, Tscene);
		newobj = obj; newscene = scene;	/*might not want to normalise*/

		int ptcount = newobj.size();
		int trailcount = 0;
		int bestcount = 0;
		int inliers_count = 0;
		Mat bestH(2, 3, CV_32F);
		vector<int> best_ind;
		double p = 0.999;	/*we want to ensure the posilibity of 'no outliner in a time' is p */
		int N = 1;		/*this is the estimated number of trails before stop*/
		double th = 1;	/*threshold for inliers and outliers*/
		double s = 3;		/*number of points to get matrix H*/
		unsigned int seed = time(NULL);
		RNG rng(seed);
		double p_no_outlier;

		while (trailcount < N) {
			trailcount++;
			inliers_count = 0;
			/*randomly sample s out of the matching pairs*/
			int* ind = new int[s];
			Mat rec(1, ptcount, CV_8UC1);
			rec.setTo(Scalar(0));
			for (int k = 0; k < s; k++) {
				ind[k] = rand() % ptcount;
				if (rec.at<uchar>(0, ind[k]) == 0)
					rec.at<uchar>(0, ind[k]) = 1;
				else
					k--;
			}

			//for (int k = 0; k < s; k++) {
			//	cout << ind[k] << endl;
			//}

			/*get affine matrix from the selected pair*/
			vector<Point2f> currentobj = { newobj[ind[0]], newobj[ind[1]], newobj[ind[2]] };
			vector<Point2f> currentscene = { newscene[ind[0]], newscene[ind[1]], newscene[ind[2]] };
			Mat currentH = getAffineTransform(currentobj, currentscene);
			//cout << currentH << endl;

			/*test the proposed affine matrix*/
			vector<Point2f> transformed_obj, transformed_scene;
			transform(newobj, transformed_obj, currentH);
			invertAffineTransform(currentH, currentH);
			transform(newscene, transformed_scene, currentH);
			vector<int> inliers_ind;
			for (int k = 0; k < ptcount; k++) {
				int nn = transformed_obj.size();
				if (norm(transformed_obj[k] - newscene[k]) < th&&norm(transformed_scene[k] - newobj[k]) < th) {
					inliers_count++;
					inliers_ind.push_back(k);
				}
			}
			if (inliers_count > bestcount) {
				bestcount = inliers_count;
				bestH = currentH;
				best_ind = inliers_ind;

				/*probability of no outliers in s samples*/
				p_no_outlier = 1 - pow(double(bestcount) / ptcount, s);
				N = log(1 - p) / log(p_no_outlier-1e-8); /*update N*/
			}
			if (trailcount > maxtrail) {
				printf("reach max trial number\n");
				waitKey();
				break;
			}
		}

		/*find affine based on the best inliers*/
		vector<Point2f> bestobj, bestscene;
		for (int k = 0; k < bestcount; k++) {
			bestobj.push_back(newobj[best_ind[k]]);
			bestscene.push_back(newscene[best_ind[k]]);
		}
		Mat finalH = mysolveAffine(bestobj, bestscene);
		//cout << Tscene << endl;
		//invert(Tscene, Tscene);
		//cout << Tscene << endl;
		//finalH = Tscene*finalH*Tobj;
		return finalH;
	}
}

//! Augment the point vector with 1's, and normalize the coordinate
void mynormalize(vector<Point2f> src, vector<Point2f> &dst, Mat &T) {
	Scalar center = mean(src);
	double meandist=0;
	double scale;
	for (int i = 0; i < src.size(); i++) {
		meandist += sqrt((src[i].x - center[1])*(src[i].x - center[1]) + (src[i].y - center[2])*(src[i].y - center[2]));
	}
	meandist = meandist / src.size();
	scale = sqrt(2) / meandist;
	
	for (int i = 0; i < src.size(); i++) {
		dst.push_back(Point2f(scale*(src[i].x - center[0]), scale*(src[i].y - center[1])));
	}

	T = (Mat_<float>(3, 3) << scale, 0, -scale*center[0], 0, scale, -scale*center[1], 0, 0, 1);
	return;
}

//! solve affine transform matrix with more than 3 pairs of point with svd
//	T*src = dst (TA=B)
//	src'*T' = dst'
//	T:2x3
Mat mysolveAffine(vector<Point2f> src, vector<Point2f> dst) {
	Mat T(2, 3, CV_32F);

	int N = src.size();
	if (src.size()!= dst.size()) {
		printf("src and dst different length!\n");
		waitKey();
		return Mat();
	}
	Mat A = Mat(src).reshape(1).t();
	Mat row = Mat::ones(1, N, CV_32F);
	A.push_back(row);
	cout << A << endl;
	Mat B = Mat(dst).reshape(1).t();
	cout << B << endl;
	solve(A.t(), B.t(), T, DECOMP_NORMAL);
	cout << T;
	T = T.t();
	return T;
}
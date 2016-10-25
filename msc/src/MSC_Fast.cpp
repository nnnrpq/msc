/*	Frame by frame optimazation is added to this version.
	Also warpperspective is replaced by calculating the mapping manually.
		- Zhangyuan Wang 06/12
*/
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include<stdlib.h>
#include<stdio.h>
#include<vector>
#include <cstdio>
#include<fstream>
#include<iostream>
#include<math.h>
//#include <unistd.h>
#include"MSC.h"
#include"Learn_Image.h"
#include "Image_PreProcessing.h"


#define PI 3.14159265

using namespace std;
using namespace cv;

int layer_count;

//Mat affine_transformation;
double MAX_VAL = 255;
//Mat transformations;
//vector < Mat > transformation_set;

/* layer control*/
bool xTranslate_layer = 1;
bool yTranslate_layer = 1;
bool rotate_layer = 1;
bool scale_layer = 1;

bool READFROMFILE =0;

/* If not read the transformation from file, test all the possible parameters*/
double xTranslate_step = 10;
double yTranslate_step = 10;
double rotate_step = 10;
double scale_step = 5;

vector<double> xT_val;
vector<double> yT_val;
vector<double> rot_val;
vector<double> sc_val;
double maxscale_para = 1;	/*maximum scaling factor*/
double minscale_para = 0.5;	/*minimum scaling factor*/

/* k-step value*/
double k_xTranslate = 0.5;
double k_yTranslate = 0.5;
double k_rotate = 0.4;
double k_scale = 0.35;
double k_memory = 0.25;

/* debug and display control*/
int test_forw = 0;
int test_back = 0;
int test_comp =0;
int dispInMid = 0;
int dispsc = 0;
int dispg = 0;

/* others*/
double *k_transformations;
Mat C = (Mat_<double>(1,3) << 0, 0, 1);
#define FORWARD 1
#define BACKWARD -1

/* scaling control*/
/* start scaling normalizer after all the rest layer are clear*/
int startscale = 1;

void getTransform(Size img_size, vector < Mat > &transformation_set, vector< Mat > &G,TransformationSet lastTr = TransformationSet()) {
	Mat affine_transformation;
	Mat transformations;
	int lcount = 0;
	bool framectl = lastTr.nonIdenticalCount != -1;

	double xTcenter = -lastTr.xTranslate;
	double yTcenter = -lastTr.yTranslate;
	double angcenter = -lastTr.theta;
	double sccenter = lastTr.scale;
	double xTrange = framectl ? 0.2*img_size.width : 0.8*img_size.width;
	double yTrange = framectl ? 0.2*img_size.height : 0.8*img_size.height;
	double rotrange = framectl ? 0.4*180 : 180;
	double scrange = framectl ? 0.4 * (maxscale_para- minscale_para) : (maxscale_para - minscale_para);
	if (framectl) {
		xTranslate_step = max(2.0, round(xTranslate_step / 2));
		yTranslate_step = max(2.0, round(yTranslate_step / 2));
		rotate_step = max(2.0, round(rotate_step / 2));
		scale_step = max(2.0, round(scale_step / 2));
	}
	if (xTranslate_layer == true) {
		double xTranslate1 = xTcenter;
		double xTranslate2 = xTcenter;
		for (int i = 0; i <= ceil(xTranslate_step / 2); i++) {
			affine_transformation = (Mat_<float>(1, 9) << 1, 0, xTranslate1, 0, 1, 0, 0, 0, 1);
			transformations.push_back(affine_transformation);
			xT_val.push_back(xTranslate1);
			xTranslate1 += xTrange / xTranslate_step;
			if (i != 0) {
				affine_transformation = (Mat_<float>(1, 9) << 1, 0, xTranslate2, 0, 1, 0, 0, 0, 1);
				transformations.push_back(affine_transformation);
				xT_val.push_back(xTranslate2);
			}
			xTranslate2 -= xTrange / xTranslate_step;
		}
		transformation_set.push_back(transformations);
		//cout << transformations << endl;
		transformations.release();
		G.push_back(Mat::ones(Size(xT_val.size(), 1), CV_32FC1));
		lcount++;
		k_transformations[lcount - 1] = k_xTranslate;
	}
	if (yTranslate_layer == true) {
		double yTranslate1 = yTcenter;
		double yTranslate2 = yTcenter;
		for (int i = 0; i <= ceil(yTranslate_step / 2); i++) {
			affine_transformation = (Mat_<float>(1, 9) << 1, 0, 0, 0, 1, yTranslate1, 0, 0, 1);
			transformations.push_back(affine_transformation);
			yT_val.push_back(yTranslate1);
			yTranslate1 += yTrange / yTranslate_step;
			if (i != 0) {
				affine_transformation = (Mat_<float>(1, 9) << 1, 0, 0, 0, 1, yTranslate2, 0, 0, 1);
				transformations.push_back(affine_transformation);
				yT_val.push_back(yTranslate2);
			}
			yTranslate2 -= yTrange / yTranslate_step;
		}
		transformation_set.push_back(transformations);
		//cout << transformations << endl;
		transformations.release();
		G.push_back(Mat::ones(Size(yT_val.size(), 1), CV_32FC1));
		lcount++;
		k_transformations[lcount - 1] = k_yTranslate;
	}
	if (rotate_layer == true) {
		double theta1 = angcenter;
		double theta2 = angcenter;
		double thetad;
		for (int i = 0; i <= ceil(rotate_step / 2); i++) {
			thetad = theta1 * PI / 180;
			affine_transformation = (Mat_<float>(1, 9) << cos(thetad), -sin(thetad), 0, sin(thetad), cos(thetad), 0, 0, 0, 1);
			transformations.push_back(affine_transformation);
			rot_val.push_back(theta1);
			if (i != 0 && i != ceil(rotate_step / 2)) {
				thetad = theta2 * PI / 180;
				affine_transformation = (Mat_<float>(1, 9) << cos(thetad), -sin(thetad), 0, sin(thetad), cos(thetad), 0, 0, 0, 1);
				transformations.push_back(affine_transformation);
				rot_val.push_back(theta2);
			}
			theta1 += rotrange / rotate_step;
			theta2 -= rotrange / rotate_step;
		}
		//cout << transformations << endl;

		transformation_set.push_back(transformations);
		transformations.release();
		G.push_back(Mat::ones(Size(rot_val.size(), 1), CV_32FC1));
		lcount++;
		k_transformations[lcount - 1] = k_rotate;
	}
	if (scale_layer == true) {
		if (!framectl) {
			double scale = 1;
			for (int i = 0; i <= scale_step; i++) {
				affine_transformation = (Mat_<float>(1, 9) << scale, 0, 0, 0, scale, 0, 0, 0, 1);
				transformations.push_back(affine_transformation);
				sc_val.push_back(scale);
				scale -= (maxscale_para - minscale_para) / scale_step;
			}
		}
		else {
			double sc1 = sccenter;
			double sc2 = sccenter;
			for (int i = 0; i <= ceil(scale_step / 2); i++) {
				affine_transformation = (Mat_<float>(1, 9) << sc1, 0, 0, 0, sc1, 0, 0, 0, 1);
				transformations.push_back(affine_transformation);
				sc_val.push_back(sc1);
				sc1 += scrange / scale_step;
				if (i != 0) {
					affine_transformation = (Mat_<float>(1, 9) << sc2, 0, 0, 0, sc2, 0, 0, 0, 1);
					transformations.push_back(affine_transformation);
					sc_val.push_back(sc2);
				}
				sc2 -= scrange / scale_step;
			}
		}
		//double scale1 = 1;
		//double newscale;
		//double temp = maxscale_para;
		//maxscale_para = 1 / minscale_para;
		//minscale_para = 1 / temp;
		//for (int i = 0; i <= scale_step; i++) {
		//	newscale = 1 / scale1;
		//	affine_transformation = (Mat_<float>(1, 9) << newscale, 0, 0, 0, newscale, 0, 0, 0, 1);
		//	transformations.push_back(affine_transformation);
		//	sc_val.push_back(newscale);
		//	scale1 += (maxscale_para - minscale_para) / scale_step;
		//}
		//cout << transformations << endl;
		transformation_set.push_back(transformations);
		transformations.release();
		G.push_back(Mat::ones(Size(sc_val.size(), 1), CV_32FC1));
		lcount++;
		k_transformations[lcount - 1] = k_scale;

		/*push back 0 for k_scale first*/
		//k_transformations[lcount - 1] = 0.01;
	}
}

/* get the mapping of pixels for all the forward transformation*/
void getTransMap(Size img_size, int flag, vector<Mat> &ResultMap) {
	int pos = 0;
	int pixcount = img_size.height*img_size.width;
	if (flag == 1) {
		/* this is forward*/
	}
	else if (flag == -1) {
		/* this is backward*/
	}
	else {
		printf("wrong flag\n"); exit(0);
	}

	if (xTranslate_layer) {
		Mat Map(xT_val.size(),pixcount, CV_32SC1);
		for (int k = 0; k < xT_val.size(); k++) {
			/* for all the xtranslations*/
			pos = 0;
			Mat temp(img_size.height, img_size.width, CV_32SC1);
			int* pt = (int*)temp.data;
			int xT = round(xT_val[k])*flag;
			for (int i = 0; i < img_size.height; i++) {
				for (int j = 0; j < img_size.width; j++) {
					if (j-xT<0 || j- xT >=img_size.width) {
						pt[pos] = -1;
					}
					else {
						pt[pos] = pos - xT;
					}
					pos++;
				}
				//cout << temp.row(0) << endl;
			}
			//cout << temp << endl;
			Mat xx = temp.reshape(1, 1);
			xx.row(0).copyTo(Map.row(k));
		}
		//cout << Map << endl;
		ResultMap.push_back(Map);
	}
	if (yTranslate_layer) {
		Mat Map(yT_val.size(), pixcount, CV_32SC1);
		for (int k = 0; k < yT_val.size(); k++) {
			/* for all the ytranslations*/
			pos = 0;
			Mat temp(img_size.height, img_size.width, CV_32SC1);
			int* pt = (int*)temp.data;
			int yT = round(yT_val[k])*flag;
			for (int i = 0; i < img_size.height; i++) {
				for (int j = 0; j < img_size.width; j++) {
					if (i < yT || i-yT>=img_size.height) {
						pt[pos] = -1;
					}
					else {
						pt[pos] = pos - img_size.width*yT;
					}
					pos++;
				}
			}
			Mat xx = temp.reshape(1, 1);
			xx.row(0).copyTo(Map.row(k));
		}
		ResultMap.push_back(Map);
	}
	if (rotate_layer) {
		Mat Map(rot_val.size(), pixcount, CV_32SC1);
		for (int k = 0; k < rot_val.size(); k++) {
			/* for all the rotation*/
			pos = 0;
			Mat temp(img_size.height, img_size.width, CV_32SC1);
			int* pt = (int*)temp.data;
			double ang = rot_val[k]*flag*PI/180;
			int x, y, newx, newy;
			double cosang, sinang;
			for (int i = 0; i < img_size.height; i++) {
				for (int j = 0; j < img_size.width; j++) {
					x = j - img_size.width / 2;
					y = i - img_size.height / 2;
					cosang = cos(ang);
					sinang = sin(ang);
					newx = cosang*x - sinang*y;
					newy = sinang*x + cosang*y;
					if (abs(newx) >= img_size.width / 2 || abs(newy)>= img_size.height / 2)
						pt[pos] = -1; 
					else
						pt[pos] = floor(newx + img_size.width / 2) + floor(newy + img_size.height / 2)*img_size.width;
					pos++;
				}
			}
			Mat xx = temp.reshape(1, 1);
			xx.row(0).copyTo(Map.row(k));
		}
		ResultMap.push_back(Map);
	}
	if (scale_layer) {
		Mat Map(sc_val.size(), pixcount, CV_32SC1);
		for (int k = 0; k < sc_val.size(); k++) {
			/* for all the rotation*/
			pos = 0;
			Mat temp(img_size.height, img_size.width, CV_32SC1);
			int* pt = (int*)temp.data;
			double sc = flag==1?sc_val[k]:1/ sc_val[k];
			int x, y, newx, newy, cosang, sinang;
			for (int i = 0; i < img_size.height; i++) {
				for (int j = 0; j < img_size.width; j++) {
					x = j - img_size.width / 2;
					y = i - img_size.height / 2;
					newx = sc*x;
					newy = sc*y;
					if (abs(newx) >= img_size.width / 2 || abs(newy)>=img_size.height / 2)
						pt[pos] = -1;
					else
						pt[pos] = floor(newx + img_size.width / 2) + floor(newy + img_size.height / 2)*img_size.width;
					pos++;
				}
			}
			Mat xx = temp.reshape(1, 1);
			xx.row(0).copyTo(Map.row(k));
			//cout << xx;
		}
		ResultMap.push_back(Map);
	}
}

int SL_MSC(Mat Input_Image, Mat Memory_Images, Size img_size, Mat *Fwd_Path, Mat *Bwd_Path, TransformationSet & finalTrans){
	Mat affine_transformation;
	double MAX_VAL = 255;
	Mat transformations;
	vector < Mat > transformation_set;

    Mat G_layer; // Competition function values for each layer.
    vector< Mat > G; // The competition function
    int iteration_count = 100; // Number of iterations for which MSC will operate.
    int ret = -1;
    double verified_ret;
    FILE *fp;
    double dot_product_input_object = Input_Image.dot(Input_Image);
    // Read all the transformations for MSC paths as well as the layer information.
    //printf("Dot product #1 done\n");
    // open a file in read mode.
    // Check whether the necessary file exists or not.
    
    // int File_exist =  access("Layer_Count.txt", F_OK);
    
    // If the file exists then read the count of memory images.
    // Else we have add the image to the memory and update the file.
    
    /*
    if (File_exist == 0) {
        fp = fopen( "Layer_Count.txt", "r" );
        fscanf(fp, "%d", &layer_count);
        fclose(fp);
    }else{
        layer_count = 0;
    }
     */
    
    int layer_count = (int)1.0+(int)(xTranslate_layer)+(int)(yTranslate_layer)+(int)(rotate_layer)+(int)(scale_layer);
     
    transformations.release();
    G_layer.release();
    
    
    k_transformations = new double[layer_count];
    
    k_transformations[layer_count-1] = k_memory;

	if (READFROMFILE) {
		for (int i = 1; i < layer_count; i++) {
			char Layer_filename[200];
			char layer_name[40];
			printf("iteration #: %d\n", i);
			strcpy(Layer_filename, "Layer_");

			if (xTranslate_layer == true) {
				strcpy(layer_name, "xTranslate_layer");
				xTranslate_layer = false;
				k_transformations[i - 1] = k_xTranslate;
			}
			else if (yTranslate_layer == true) {
				strcpy(layer_name, "yTranslate_layer");
				yTranslate_layer = false;
				k_transformations[i - 1] = k_yTranslate;
			}
			else if (rotate_layer == true) {
				strcpy(layer_name, "Rotate_layer");
				rotate_layer = false;
				k_transformations[i - 1] = k_rotate;
			}
			else if (scale_layer == true) {
				strcpy(layer_name, "scale_layer");
				k_transformations[i - 1] = k_scale;
			}

			strcat(Layer_filename, layer_name);
			strcat(Layer_filename, ".txt");
			fp = fopen(Layer_filename, "r+");
			float next = 0;
			int count = 0;
			while (fscanf(fp, "%f ", &next) > 0) // parse %f followed by ' '
			{
				affine_transformation.push_back(next);
				count++;
				if (count == 9) {
					transformations.push_back(affine_transformation.reshape(0, 1));
					G_layer.push_back(1);
					affine_transformation.release();
					count = 0;
				}
			}

			fclose(fp);
			transformation_set.push_back(transformations);
			if (i < layer_count) {
				G.push_back(G_layer.reshape(0, 1));
			}
			//cout<<"Transformations"<<endl;
			//cout<<transformations<<endl;
			G_layer.release();
			transformations.release();
		}
	}
	else 
		getTransform(img_size, transformation_set, G, finalTrans);

	/* get the mapping for transformation*/
	vector<Mat> MapForw, MapBack;
	getTransMap(img_size, FORWARD, MapForw);
	getTransMap(img_size, BACKWARD, MapBack);

	for (int i = 0; i < Memory_Images.rows; i++) {
		G_layer.push_back(1);
	}
	G.push_back(G_layer.reshape(0, 1));
	G_layer.release();

	int* idxTrans = new int[layer_count - 1];

	int count = 0;
    while(iteration_count > 0){
        iteration_count--;
        printf("About to call MSC %d\n",count++);
        ret = MapSeekingCircuit(Input_Image, Memory_Images, img_size, Fwd_Path, Bwd_Path, layer_count, MapForw,MapBack, &G, k_transformations);
        
		bool flag = 1;		/* 1 for stopping the msc*/
        if(iteration_count %5 == 0){
			///* only inspect before the scaling layer*/
			//for (int kk = 0; kk < 2; kk++) {
			//	cout << "-------\n"<<G[kk] << "-------\n";
			//	if (countNonZero(G[kk]) != 1) {
			//		flag = 0;
			//		break;
			//	}
			//	else {
			//		vector<Point> idx;
			//		Mat current;
			//		G[kk].convertTo(current, CV_8UC1,100);
			//		//cout << current << endl << G[kk] << endl;
			//		findNonZero(current, idx);
			//		idxTrans[kk] = (idx[0]).x;
			//	}
			//}

 		//	if (flag) {
 		//		k_transformations[layer_count - 2] = k_scale;
			//}

			for (int kk = 0; kk < G.size() - 1; kk++) {
				cout << "-------\n" << G[kk] << "-------\n";
				if (countNonZero(G[kk]) != 1) {
					flag = 0;
					break;
				}
				else {
					vector<Point> idx;
					Mat current;
					G[kk].convertTo(current, CV_8UC1, 100);
					//cout << current << endl << G[kk] << endl;
					findNonZero(current, idx);
					idxTrans[kk] = (idx[0]).x;
				}
			}


			if (dispInMid) {
				imshow("FPV_Forward[1]", (*Fwd_Path) * 255);

				imshow("BPV[1]", (*Bwd_Path) * 255);
				cvWaitKey(0);
			}
			/* stop iteration condition: only one transformation is left*/
			/* record the final transformation*/
			if (flag) {
				double xT = -xT_val[idxTrans[0]];
				double yT = -yT_val[idxTrans[1]];
				double ang = -rot_val[idxTrans[2]];
				double sc;
				if (scale_layer)
					sc = sc_val[idxTrans[3]];
				else
					sc = 1;
				//double xT = 0;
				//double yT = 0;
				//double ang = 0;
				//double sc = 1;
				finalTrans = TransformationSet(xT, yT, ang, sc);
				break;
			}
        }
        //printf("MSC dot products are done\n");
        verified_ret = Verify_Object(Input_Image, *Bwd_Path, dot_product_input_object);
        /*
        if(verified_ret == 0){
            printf("Image not recognized\n");
        }else{
            printf("Everything seems to be fine\n");
        }
         */
    }
	ret = MapSeekingCircuit(Input_Image, Memory_Images, img_size, Fwd_Path, Bwd_Path, layer_count, MapForw, MapBack, &G, k_transformations);
	printf("The value of verified_ret is %g\n", verified_ret);
	xT_val.clear();
	yT_val.clear();
	rot_val.clear();
	sc_val.clear();
    return ret;
    
}

int MapSeekingCircuit(Mat Input_Image, Mat Memory_Images, Size img_size, Mat *Fwd_Path, Mat *Bwd_Path, int layers, vector< Mat > MapForw, vector<Mat> MapBack, vector< Mat > *G, double k_transformations[]){
    
    vector< Mat > g = *G;
    
    Fwd_Path_Values *FPV = new Fwd_Path_Values[layers];
    
    Mat *BPV = new Mat[layers];

	Mat *TranSc = new Mat_<double>[layers];
    
    /*
     The transformation matrix in openCV looks like:
     
     [                       |                   |
           xscale*cos(theta) |    -sin(theta)    |  x-translation
            sin(theta)       |  yscale*cos(theta)|  y-translation
                 0           |        0          |    1
     ]
     */
    
    FPV[0].Fwd_Superposition = Input_Image.clone();
    
    //printf("Backward path superposition \n");
    BPV[layers-1] = Superimpose_Memory_Images(Memory_Images, g[layers-1], img_size.height).clone();
    
    
    
    //imshow("BPV[layers-1]", BPV[layers-1]*255);
    if(layers>1){
        //printf("Apply transformations\n");
        for(int i = 1; i < layers; i++){
            // Perform all of the forward path transformations
            FPV[i] = ForwardTransform(FPV[i-1].Fwd_Superposition, MapForw[i-1].clone(), g[i-1],TranSc[i-1]);
            
            // Perform all of the backward path transformations
            BPV[layers-1-i] = BackwardTransform(BPV[layers-i], MapBack[layers - 1 - i].clone(), g[layers-1-i]);
        }
        
        //printf("Update competition function\n");
        for(int i = 1; i < layers; i++){
            // Update competition
            g[i-1] = UpdateCompetition(FPV[i].Transformed_Templates, BPV[i], g[i-1], img_size.height, k_transformations[i-1], TranSc[i - 1]).clone();
            
			if (dispg)
				cout<<"g"<<i-1<<"="<<g[i-1]<<endl;
            //cout<<endl;
        }
        
    }
    //cout<<"layers-1  "<<layers-1<<endl;
    g[layers-1] = UpdateCompetition_Memory(Memory_Images, FPV[layers-1].Fwd_Superposition, g[layers-1], img_size.height, k_transformations[layers-1]).clone();
    
    //cout<<"g memory: "<<g[layers-1]<<endl;
    //
 //   imshow("FPV_Forward[1]", (FPV[layers-1].Fwd_Superposition)*255);
 //   
 //   //imshow("BPV[1]", (BPV[0])*255);
	//waitKey();
 //   
    *Fwd_Path = FPV[layers-1].Fwd_Superposition;
    *Bwd_Path = BPV[0];
    
    *G = g;
    return 0;
}


Fwd_Path_Values ForwardTransform(Mat In, Mat transMap, Mat g, Mat &Transc){
    Fwd_Path_Values FPV_return;
    Mat SuperPosition;
    Mat TransformedTemplates;
    double Thresh_VAL = 100;
    float sine;
    float cosine;
    float angle;
    double matrix_determinant;
    Mat rotation_matrix;
    int count = g.cols;
    g.convertTo(g,CV_32F);

	Mat temp;

    //dst = In.clone();
    SuperPosition = Mat::zeros(In.rows, In.cols, CV_32FC1);
 //   SuperPosition = g.at<float>(0,0)*In.clone();
	//SuperPosition.convertTo(SuperPosition, CV_32F);
 //   FPV_return.Transformed_Templates.push_back(SuperPosition.reshape(0,1));
	//Transc.push_back(1.0);

	//if (test_forw) {
	//	imshow("In0", g.at<float>(0, 0)*In * 255);
	//	imshow("dst0", g.at<float>(0, 0)*dst * 255);
	//	cvWaitKey();
	//}

    for(int i=0; i<count; i++){
		Mat dst(In.rows, In.cols, CV_32F);
		if (g.at<float>(0, i) == 0) {
			Mat temp1 = Mat::zeros(Size((In.rows)*(In.cols),1), CV_32F);
			FPV_return.Transformed_Templates.push_back(temp1);
			Transc.push_back(1.0);
			continue;
		}

		//cout << transMap << endl;
		float* ptdst = (float*)dst.data;
		float* ptIn = (float*)In.data;
		int* ptidx = transMap.ptr<int>(i);
		for (int n = 0; n < In.rows*In.cols; n++) {
			if (ptidx[n] == -1)
				ptdst[n] = 0;
			else
				ptdst[n] = ptIn[ptidx[n]];
		}
 //       Mat Perspective_Transformation_Matrix_2D = Perspective_Transformation_Matrix.row(i).reshape(0,3);
 //       
 //       sine = -Perspective_Transformation_Matrix_2D.at<float>(0,1);
 //       cosine = Perspective_Transformation_Matrix_2D.at<float>(0,0);
 //       
 //       angle = atan(sine/cosine)*180/PI;
 //       
 //       matrix_determinant = (sqrt(determinant(Perspective_Transformation_Matrix_2D)));
	//	//Transc.push_back(matrix_determinant);
	//	matrix_determinant = 1 / matrix_determinant;
	////cout<<"Perspective Matrix Forward: "<<Perspective_Transformation_Matrix_2D<<endl;
 //       //cout<<"Matrix D Forward   "<<matrix_determinant<<endl;
	//	//if (0) {
 //       if(abs(angle) >= 0.0001 || abs(matrix_determinant - 1) >= 0.0001){
 //           //cout<<"Perspective Matrix Forward: "<<Perspective_Transformation_Matrix_2D<<endl;
 //           Point2f src_center(In.cols/2.0F, In.rows/2.0F);
 //           rotation_matrix = getRotationMatrix2D(src_center, angle, matrix_determinant);
 //           vconcat(rotation_matrix, C, rotation_matrix);
 //           warpPerspective( In, dst, rotation_matrix, dst.size(), INTER_NEAREST);
	//		Transc.push_back(sqrt(double(countNonZero(In)) / double(countNonZero(dst))));
	//		//imshow("In", In);
	//		//imshow("dst", dst);
	//		//cvWaitKey();
 //       }else{
 //           warpPerspective( In, dst, Perspective_Transformation_Matrix_2D, dst.size() , INTER_NEAREST);
	//		Transc.push_back(1.0);
 //       }

		Transc.push_back(sqrt(double(countNonZero(In)) / double(countNonZero(dst))));
		//Transc.push_back(1.0);

		if (0) {
			dst.convertTo(temp, CV_8U, 255);
			imshow("temp", temp); waitKey();
			//cout << temp;
			temp = CannyThreshold(temp,50, 100);
			imshow("tempafter", temp); waitKey();
			//cout << temp;
			//threshold(temp, temp, Thresh_VAL, MAX_VAL, THRESH_BINARY);
			temp.clone().convertTo(dst, CV_32F, 1.0/255);
		}
		if (test_forw) {
			imshow("In1", g.at<float>(0, i)*In * 255);
			imshow("dst1", g.at<float>(0, i)*dst * 255);
			//cout << "g=" << g.at<float>(0, i) << " angle=" << angle << " scale=" << matrix_determinant << "\n" << Perspective_Transformation_Matrix_2D << endl;
			cvWaitKey();
		}
        /*
        if(g.at<double>(0,i) < 0.3){
            g.at<double>(0,i) = 0;
        }
         */
		dst.convertTo(dst, CV_32FC1);
        Mat dst_scaled = g.at<float>(0,i)*dst;
        SuperPosition = SuperPosition + dst_scaled;

        FPV_return.Transformed_Templates.push_back(dst_scaled.reshape(0,1));
        dst.release();
    }
    //SuperPosition.convertTo(SuperPosition,CV_8U);
    threshold(SuperPosition, SuperPosition, Thresh_VAL, MAX_VAL, THRESH_TRUNC);
    FPV_return.Fwd_Superposition = SuperPosition.clone();
    return FPV_return;
}


Mat BackwardTransform(Mat In, Mat transMap, Mat g){
    Mat BPV_return;
    Mat SuperPosition;
    Mat TransformedTemplates;
    float sine;
    double Thresh_VAL = 1;
    float cosine;
    float angle;
    double matrix_determinant;
    Mat rotation_matrix;
    int count = g.cols;
    g.convertTo(g,CV_32F);

    SuperPosition = Mat::zeros(In.rows, In.cols, CV_32FC1);
    //SuperPosition = g.at<float>(0,0)*In.clone();
	//SuperPosition.convertTo(SuperPosition, CV_32F);

    for(int i=0; i<count; i++){
		Mat dst(In.rows, In.cols, CV_32F);
		if (g.at<float>(0, i) == 0) {
			continue;
		}

		float* ptdst = (float*)dst.data;
		float* ptIn = (float*)In.data;
		int* ptidx = transMap.ptr<int>(i);
		for (int n = 0; n < In.rows*In.cols; n++) {
			if (ptidx[n] == -1)
				ptdst[n] = 0;
			else
				ptdst[n] = ptIn[ptidx[n]];
		}

		if (test_back) {
			imshow("In", g.at<float>(0, i)*In * 255);
			imshow("dst", g.at<float>(0, i)*dst * 255);
			//cout << "g="<< g.at<float>(0, i)<<" angle=" << angle << " scale=" << matrix_determinant << "\n" << Perspective_Transformation_Matrix_2D << endl;
			cvWaitKey();
		}

		dst.convertTo(dst, CV_32FC1);
		
        Mat dst_scaled = g.at<float>(0,i)*dst;
        SuperPosition = SuperPosition + dst_scaled;
        dst.release();
    }
    //SuperPosition.convertTo(SuperPosition,CV_32FC1);
    threshold(SuperPosition, SuperPosition, Thresh_VAL, MAX_VAL, THRESH_TRUNC);
    //Mat Superposition_image_changed = foregroundBackgroundImageChange(SuperPosition);
    BPV_return = SuperPosition.clone();
    return BPV_return;
}

Mat Superimpose_Memory_Images(Mat M, Mat g, int r)
{
    g.convertTo(g,CV_64F);
    double Thresh_VAL = 1;
    //cout<<"Superposition of memory images, g: "<<g<<endl;
    Mat Superimposed_Image = Mat::zeros(1, M.cols, CV_64FC1);
    int row_count = M.rows;
    M.convertTo(M,CV_64FC1);
    //printf("Cols count is: %d\n", M.cols);
    for(int i=0; i < row_count; i++){
        //printf("columns in M[i] %d\n",M.row(i).cols);
        Superimposed_Image = Superimposed_Image + g.at<double>(0, i)*M.row(i);
    }
    Superimposed_Image.convertTo(Superimposed_Image,CV_32FC1);
    //printf("Memory superposition\n");
    threshold(Superimposed_Image, Superimposed_Image, Thresh_VAL, MAX_VAL, THRESH_TRUNC);
    return Superimposed_Image.reshape(0,r);
}

Mat UpdateCompetition(Mat Transformed_Templates, Mat BackwardTransform, Mat g, int r, double k, Mat TranSc,double p){
    int count = Transformed_Templates.rows;
    g.convertTo(g,CV_64F);
    Mat subtracted_g(g.rows, g.cols, CV_64FC1);
    Mat thresholded_g(g.rows, g.cols, CV_32FC1);
    double Thresh_VAL = 0.1;
    double MAX_VAL = 1;
    Mat q(g.rows, g.cols, CV_64F);
    double T_L2;
    double BackwardTransform_L2;
    double min, max;

	if (dispsc)
		cout << TranSc << endl;
    for(int i=0; i<count; i++){
		if (g.at<double>(0, i) == 0) {
			q.at<double>(0, i) = 0;
			continue;
		}
        Mat T = Transformed_Templates.row(i).reshape(0,r);
        T.convertTo(T,CV_32FC1);
        T_L2 = norm(T, NORM_L2);
		//T_L2 = sum(T)[0];
        //BackwardTransform_L2 = norm(BackwardTransform, NORM_L2);
		BackwardTransform_L2 = sum(BackwardTransform)[0];
        
		if (test_comp) {
			imshow("T", T);
			imshow("BackwardTransform", BackwardTransform);
			waitKey();
		}

		//cout << TranSc << endl;
        if(BackwardTransform_L2 !=0 && T_L2 != 0){
			if (startscale)
				q.at<double>(0,i) = T.dot(BackwardTransform)*((TranSc.at<double>(i,0)));
			//q.at<double>(0, i) = T.dot(BackwardTransform) / T_L2;
			else
				q.at<double>(0, i) = T.dot(BackwardTransform);
        }else{
            q.at<double>(0,i) = 0;
        }
		if (startscale)
			if (TranSc.at<double>(i, 0) < 1)
				p = 1.5;

    }

    //cout<<"q: "<<q<<endl;
    minMaxLoc(q, &min, &max);
   // cout<<"q_min:"<<min<<"  q_max: "<<max<<endl;
	Mat temp;
	pow(1- q / max, p, temp);
    subtract(g, k*(temp), subtracted_g) ;
	//cout << "g:" << g << "  subtracted_g: " << subtracted_g << endl;
    subtracted_g.convertTo(subtracted_g,CV_32F);
    threshold(subtracted_g, thresholded_g, Thresh_VAL, MAX_VAL, THRESH_TOZERO);
    return thresholded_g;
}

Mat UpdateCompetition_Memory(Mat Transformed_Templates, Mat BackwardTransform, Mat g, int r, double k){
    int count = Transformed_Templates.rows;
    g.convertTo(g,CV_64F);
    Mat subtracted_g(g.rows, g.cols, CV_64FC1);
    Mat thresholded_g(g.rows, g.cols, CV_32FC1);
    double Thresh_VAL = 0.3;
    Mat q(g.rows, g.cols, CV_64F);
    double min, max;
    double T_L2;
    double BackwardTransform_L2;
    //printf("About to call dot in MSC\n");
    //cout<<"g_memory: "<<g<<endl;
    for(int i=0; i<count; i++){
        Mat T = g.at<double>(0,i)*Transformed_Templates.row(i).reshape(0,r);
        T.convertTo(T,CV_32FC1);
        BackwardTransform.convertTo(BackwardTransform, CV_32FC1);
        //cvWaitKey(0);
        T_L2 = norm(T, NORM_L2);
        BackwardTransform_L2 = norm(BackwardTransform, NORM_L2);
        
        
        if(BackwardTransform_L2 !=0 && T_L2 != 0){
            q.at<double>(0,i) = T.dot(BackwardTransform)/(T_L2*BackwardTransform_L2);
        }else{
            q.at<double>(0,i) = 0;
        }
        //printf("Dot product done for iteration: %d\n", i);
    }
    //printf("Dot product has been completed\n");
    minMaxLoc(q, &min, &max);
    //cout<<"in update MEMORY q_min:"<<min<<"  q_max: "<<max<<endl;
    subtract(g, k*(1-q/max), subtracted_g);
    //cout<<"q: "<<q<<endl;
    subtracted_g.convertTo(subtracted_g,CV_32FC1);
    threshold(subtracted_g, thresholded_g, Thresh_VAL, MAX_VAL, THRESH_TOZERO);
    return thresholded_g;
}
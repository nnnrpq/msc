/* The latest version, with experiment of openmp which doesn't work well because of the warpaffine parallel approach
	(openmp parallel cannot be overlapped).
	k value is tuned to be larger.

	- Zhangyuan Wang 06/12/2016
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
#include <ctime>

#include <pthread.h>
#pragma comment(lib,"pthreadVC2.lib")

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
double minscale_para = 0.3;	/*minimum scaling factor*/

/* k-step value*/
double k_xTranslate = 0.8;
double k_yTranslate = 0.8;
double k_rotate = 0.4;
double k_scale = 0.4;
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

/* FPV and BPV are global variable to reduce memory allocation*/
Fwd_Path_Values *FPV;
Mat *BPV;

/* scaling control*/
/* start scaling normalizer after all the rest layer are clear*/
int startscale = 1;

uint32_t t_total, t_loop;

/* get the transformation set used for current MSC session*/
void getTransform(Size img_size, vector < Mat > &transformation_set, vector< Mat > &G, TransformationSet lastTr = TransformationSet());

/* get the mapping of pixels for all the forward transformation*/
void getTransMap(Size img_size, int flag, vector<Mat> &ResultMap);

int SL_MSC(Mat Input_Image, Mat Memory_Images, Size img_size, Mat *Fwd_Path, Mat *Bwd_Path, TransformationSet & finalTrans){
	//imshow("get img", Memory_Images);
	//waitKey();

	t_total = clock();
	t_loop = 0;

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

    
    int layer_count = 1+(int)(xTranslate_layer)+(int)(yTranslate_layer)+(int)(rotate_layer)+(int)(scale_layer);
    
    k_transformations = new double[layer_count-1];
    
	if (finalTrans.nonIdenticalCount != -1) {
		cout << "quick solution by frame optimization" << endl;
	}
	
	getTransform(img_size, transformation_set, G, finalTrans);

	
	/* get the mapping for transformation,
		not used any more*/

	//vector<Mat> MapForw, MapBack;
	//MapForw.reserve(layer_count - 1);
	//MapBack.reserve(layer_count - 1);
	//getTransMap(img_size, FORWARD, MapForw);
	//getTransMap(img_size, BACKWARD, MapBack);



	int* idxTrans = new int[layer_count - 1];

	/* Pre allocation of memory for FPV and BPV */
	FPV = new Fwd_Path_Values[layer_count];
	FPV[0].Fwd_Superposition = Input_Image.clone();
	for (int i = 1; i < layer_count; i++) {
		FPV[i].Fwd_Superposition = Mat::zeros(Input_Image.rows, Input_Image.cols, CV_32F);
		FPV[i].Transformed_Templates = Mat::zeros(Size(Input_Image.rows*Input_Image.cols, G[i-1].cols), CV_32F);
	}
	BPV = new Mat[layer_count];
	for (int i = 0; i < layer_count; i++) {
		BPV[i] = Mat::zeros(Input_Image.rows, Input_Image.cols, CV_32F);
	}


    while(iteration_count > 0){
        iteration_count--;
        //printf("About to call MSC %d\n",count++);
        ret = MapSeekingCircuit(Input_Image, Memory_Images, img_size, Fwd_Path, Bwd_Path, layer_count, transformation_set, &G, k_transformations);
        
		/* check the g value every 5 iterations*/
		bool flag = 1;		/* 1 for stopping the msc*/
        if(iteration_count %5 == 0){
			for (int kk = 0; kk < G.size(); kk++) {
				//cout << "-------\n" << G[kk] << "-------\n";
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

			/* display the result every iteration*/
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
				finalTrans = TransformationSet(xT, yT, ang, sc);
				break;
			}
        }
        //printf("MSC dot products are done\n");
        verified_ret = Verify_Object(Input_Image, *Bwd_Path, dot_product_input_object);
    }
	ret = MapSeekingCircuit(Input_Image, Memory_Images, img_size, Fwd_Path, Bwd_Path, layer_count, transformation_set, &G, k_transformations);

	/* clear memory for vector*/
	vector<double>().swap(xT_val);
	vector<double>().swap(yT_val);
	vector<double>().swap(rot_val);
	vector<double>().swap(sc_val);
	vector<Mat>().swap(G);

	t_total = clock()-t_total;

	printf("it takes %d/%d for loop\n", t_loop, t_total);
    return ret;
    
}

int MapSeekingCircuit(Mat Input_Image, Mat Memory_Images, Size img_size, Mat *Fwd_Path, Mat *Bwd_Path, int layers, vector< Mat > image_transformations, vector< Mat > *G, double k_transformations[]){
    
    vector< Mat > g = *G;
	Mat *TranSc = new Mat_<float>[layers];
    /*
     The transformation matrix in openCV looks like:
     
     [                       |                   |
           xscale*cos(theta) |    -sin(theta)    |  x-translation
            sin(theta)       |  yscale*cos(theta)|  y-translation
                 0           |        0          |    1
     ]
     */
   
	threshold(Memory_Images, BPV[layers - 1], 1, MAX_VAL, THRESH_TRUNC);
	BPV[layers - 1].convertTo(BPV[layers - 1], CV_32FC1);
    
    
	if (layers > 1) {
		/* Try out openmp parallel section here, to parallel the forward and backwawrd.
			But it didn't work well, probably because of the already parallel approach in warpaffine*/
#pragma omp parallel sections
	{
#pragma omp section
		for (int i = 1; i < layers; i++) {
			// Perform all of the forward path transformations
			TranSc[i - 1] = Mat(Size(g[i - 1].cols, 1), CV_32F);
			FwtArg fwrd;
			fwrd.pg = &g[i - 1];
			fwrd.pIn = &FPV[i - 1].Fwd_Superposition;
			fwrd.pInFP = &FPV[i];
			fwrd.pTransc = &TranSc[i - 1];
			fwrd.ptrans = &(image_transformations[i - 1]);
			fwrd.ret = &FPV[i];
			pthread_t tid1;
			void *status1;
			//pthread_create(&tid1, NULL, ForwardTransform, &fwrd);
			ForwardTransform(&fwrd);
		}
#pragma omp section
		for (int i = 1; i < layers; i++) {
			BktArg bckwrd;
			bckwrd.pg = &g[layers - 1 - i];
			bckwrd.pIn = &BPV[layers - i];
			bckwrd.ptrans = &(image_transformations[layers - i - 1]);
			bckwrd.ret = &BPV[layers - 1 - i];
			pthread_t tid2;
			void *status2;
			//pthread_create(&tid2, NULL, BackwardTransform, &bckwrd);
			BackwardTransform(&bckwrd);

		}
	}
		

	/* This parallel for should help.*/
#pragma omp parallel for
        for(int i = 1; i < layers; i++){
            // Update competition
            g[i-1] = UpdateCompetition(FPV[i].Transformed_Templates, BPV[i], g[i-1], img_size.height, k_transformations[i-1], TranSc[i - 1]).clone();
			if (dispg)
				cout<<"g"<<i-1<<"="<<g[i-1]<<endl;
            //cout<<endl;
        }
        
    }

    *Fwd_Path = FPV[layers-1].Fwd_Superposition.clone();
    *Bwd_Path = BPV[0].clone();
    
    *G = g;

	/* Release memory*/
	for (int i = 0; i < layers; i++) {
		TranSc[i].release();
	}

	
    return 0;
}

/* The function interface is writen by passing only one struture pointer, 
which include the pointer to input and output*/
void *ForwardTransform(void* pargin) {
/* parse the input/output*/
	FwtArg* argin = (FwtArg*)pargin;
	Mat In = *argin->pIn;
	Fwd_Path_Values* InFP = argin->pInFP;
	Mat Perspective_Transformation_Matrix = (*argin->ptrans).clone();
	Mat g = *argin->pg;
	Mat Transc = *argin->pTransc;
	Fwd_Path_Values* FPV_return = argin->ret;

	Mat SuperPosition;
	Mat TransformedTemplates;
	double Thresh_VAL = 100;
	float sine;
	float cosine;
	float angle;
	double matrix_determinant;

	int count = g.cols;
	g.convertTo(g, CV_32F);

	Mat temp;
	//Mat retTemp = Mat::zeros(Size(In.rows*In.cols,count),CV_32F);
	Mat& retTemp = InFP->Transformed_Templates;

	//dst = In.clone();
	SuperPosition = (InFP->Fwd_Superposition).setTo(0);

	/* This doesn't help a lot*/
//#pragma omp parallel for
	for (int i = 0; i<count; i++) {
		Mat rotation_matrix;
		Mat dst(In.rows, In.cols, CV_32F);

		if (g.at<float>(0, i) == 0) {		/* skip those zero weight transformation*/
			Transc.at<float>(0, i) = (1.0);
			continue;
		}

		//cout << transMap << endl;
		uint32_t t_temp = clock();
		Mat Perspective_Transformation_Matrix_2D = Perspective_Transformation_Matrix.row(i).reshape(0, 2);

		sine = -Perspective_Transformation_Matrix_2D.at<float>(0, 1);
		cosine = Perspective_Transformation_Matrix_2D.at<float>(0, 0);
		angle = roundf(atan(sine / cosine) * 180 / PI);
		matrix_determinant = (sqrt(Perspective_Transformation_Matrix_2D.at<float>(0, 0)*Perspective_Transformation_Matrix_2D.at<float>(1, 1) - Perspective_Transformation_Matrix_2D.at<float>(0, 1)*Perspective_Transformation_Matrix_2D.at<float>(1, 0)));
		matrix_determinant = 1 / matrix_determinant;

		if (abs(angle) >= 0.0001 || abs(matrix_determinant - 1) >= 0.0001) {
			Point2f src_center(In.cols / 2.0F, In.rows / 2.0F);
			rotation_matrix = getRotationMatrix2D(src_center, angle, matrix_determinant);
			warpAffine(In, dst, rotation_matrix, dst.size(), INTER_NEAREST);

			Transc.at<float>(0, i) = (sqrt(double(countNonZero(In)) / double(countNonZero(dst))));
		}
		else {
			warpAffine(In, dst, Perspective_Transformation_Matrix_2D, dst.size(), INTER_NEAREST);
			Transc.at<float>(0, i) = (1.0);
		}

		//#pragma omp critical
		t_loop += clock() - t_temp;

		if (test_forw) {
			imshow("In1", g.at<float>(0, i)*In * 255);
			imshow("dst1", g.at<float>(0, i)*dst * 255);
			//cout << "g=" << g.at<float>(0, i) << " angle=" << angle << " scale=" << matrix_determinant << "\n" << Perspective_Transformation_Matrix_2D << endl;
			cvWaitKey();
		}

		dst.convertTo(dst, CV_32FC1);
		dst*= g.at<float>(0, i);		/* This could save a one-time memory allocation*/

#pragma omp critical
		SuperPosition = SuperPosition + dst;
		dst.reshape(0, 1).copyTo(retTemp.row(i));
		dst.release();
	}

	/* parse output*/
	FPV_return->Transformed_Templates = retTemp;
	threshold(SuperPosition, SuperPosition, Thresh_VAL, MAX_VAL, THRESH_TRUNC);
	FPV_return->Fwd_Superposition = SuperPosition.clone();

	return 0;
}

/* Similar to the Forward*/
void *BackwardTransform(void* pArgin) {
	BktArg* argin = (BktArg*)pArgin;
	Mat In = *argin->pIn;
	Mat Perspective_Transformation_Matrix = (*argin->ptrans).clone();
	Mat g = *argin->pg;
	Mat* BPV_return = argin->ret;

	Mat& SuperPosition = *argin->ret;
	SuperPosition.setTo(0);
	Mat TransformedTemplates;
	float sine;
	double Thresh_VAL = 1;
	float cosine;
	float angle;
	double matrix_determinant;
	//Mat rotation_matrix;
	int count = g.cols;
	g.convertTo(g, CV_32F);

	//SuperPosition = Mat::zeros(In.rows, In.cols, CV_32FC1);
	//SuperPosition = g.at<float>(0,0)*In.clone();
	//SuperPosition.convertTo(SuperPosition, CV_32F);

//#pragma omp parallel for 
	for (int i = 0; i<count; i++) {
		Mat rotation_matrix;
		Mat dst(In.rows, In.cols, CV_32F);
		if (g.at<float>(0, i) == 0) {
			continue;
		}

		uint32_t t_temp = clock();
		Mat Perspective_Transformation_Matrix_2D = Perspective_Transformation_Matrix.row(i).reshape(0, 2);

		sine = -Perspective_Transformation_Matrix_2D.at<float>(0, 1);
		cosine = Perspective_Transformation_Matrix_2D.at<float>(0, 0);

		angle = (-1)*roundf(atan(sine / cosine) * 180 / PI);

		matrix_determinant = (sqrt(Perspective_Transformation_Matrix_2D.at<float>(0, 0)*Perspective_Transformation_Matrix_2D.at<float>(1, 1) - Perspective_Transformation_Matrix_2D.at<float>(0, 1)*Perspective_Transformation_Matrix_2D.at<float>(1, 0)));
		//cout<<"Perspective matrix backward :"<<Perspective_Transformation_Matrix_2D<<endl;
		//cout<<"Matrix D Backward   "<<matrix_determinant<<endl;

		//if (0) {
		if (abs(angle) >= 0.0001 || abs(matrix_determinant - 1) >= 0.0001) {
			//cout<<"Perspective matrix backward :"<<Perspective_Transformation_Matrix_2D<<endl;
			Point2f src_center(In.cols / 2.0F, In.rows / 2.0F);
			rotation_matrix = getRotationMatrix2D(src_center, angle, matrix_determinant);
			//vconcat(rotation_matrix, C, rotation_matrix);
			warpAffine(In, dst, rotation_matrix, dst.size(), INTER_NEAREST);
		}
		else {
			Perspective_Transformation_Matrix_2D.at<float>(0, 2) = (-1)*Perspective_Transformation_Matrix_2D.at<float>(0, 2);
			Perspective_Transformation_Matrix_2D.at<float>(1, 2) = (-1)*Perspective_Transformation_Matrix_2D.at<float>(1, 2);
			warpAffine(In, dst, Perspective_Transformation_Matrix_2D, dst.size(), INTER_NEAREST);
		}
//
//#pragma omp critical
		t_loop += clock() - t_temp;

		if (test_back) {
			imshow("In", g.at<float>(0, i)*In * 255);
			imshow("dst", g.at<float>(0, i)*dst * 255);
			//cout << "g="<< g.at<float>(0, i)<<" angle=" << angle << " scale=" << matrix_determinant << "\n" << Perspective_Transformation_Matrix_2D << endl;
			cvWaitKey();
		}

		dst.convertTo(dst, CV_32FC1);

		dst*= g.at<float>(0, i);

#pragma omp critical
		SuperPosition = SuperPosition + dst;

		dst.release();
	}
	//SuperPosition.convertTo(SuperPosition,CV_32FC1);
	threshold(SuperPosition, SuperPosition, Thresh_VAL, MAX_VAL, THRESH_TRUNC);
	//Mat Superposition_image_changed = foregroundBackgroundImageChange(SuperPosition);
	//BPV_return = &(SuperPosition);
	//SuperPosition.copyTo(*BPV_return);
	//BPV_return = &SuperPosition.clone();

	return 0;
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
    g.convertTo(g,CV_32F);
    Mat subtracted_g(g.rows, g.cols, CV_64FC1);
    Mat thresholded_g(g.rows, g.cols, CV_32FC1);
    double Thresh_VAL = 0.1;
    double MAX_VAL = 1;
    Mat q(g.rows, g.cols, CV_32F);
    double T_L2;
    double BackwardTransform_L2;
    double min, max;

	if (dispsc)
		cout << TranSc << endl;
    for(int i=0; i<count; i++){
		if (g.at<float>(0, i) == 0) {
			q.at<float>(0, i) = 0;
			continue;
		}
        Mat T = Transformed_Templates.row(i).reshape(0,r);
        T.convertTo(T,CV_32FC1);
        //T_L2 = norm(T, NORM_L2);
		T_L2 = sum(T)[0];
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
				/*q.at<float>(0,i) = T.dot(BackwardTransform)*(pow(TranSc.at<float>(0,i),0.8));*/
				q.at<float>(0, i) = T.dot(BackwardTransform)*(TranSc.at<float>(0, i));
			//q.at<double>(0, i) = T.dot(BackwardTransform) / T_L2;
			else
				q.at<float>(0, i) = T.dot(BackwardTransform);
        }else{
            q.at<float>(0,i) = 0;
        }
		if (startscale)
			if (TranSc.at<float>(0,i) < 1)
				p = 1;

    }

	minMaxLoc(q, &min, &max);
	Mat temp;
	pow(1- q / max, p, temp);			/* The p value here is to be determined. Currently it is set to 1.*/
    subtract(g, k*(temp), subtracted_g) ;
    subtracted_g.convertTo(subtracted_g,CV_32F);
    threshold(subtracted_g, thresholded_g, Thresh_VAL, MAX_VAL, THRESH_TOZERO);
    return thresholded_g;
}

/* Not used any more*/
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

void getTransform(Size img_size, vector < Mat > &transformation_set, vector< Mat > &G, TransformationSet lastTr)
{
	Mat affine_transformation;
	Mat transformations;
	int lcount = 0;
	bool framectl = lastTr.nonIdenticalCount != -1;

	double xTcenter = -lastTr.xTranslate;
	double yTcenter = -lastTr.yTranslate;
	double angcenter = -lastTr.theta;
	double sccenter = max(lastTr.scale, 0.4);
	double xTrange = framectl ? 0.2*img_size.width : 0.8*img_size.width;
	double yTrange = framectl ? 0.2*img_size.height : 0.8*img_size.height;
	double rotrange = framectl ? 0.4 * 180 : 180;
	double scrange = framectl ? 0.4 * (maxscale_para - minscale_para) : (maxscale_para - minscale_para);



	double steps = 3.0;
	if (framectl) {
		xTranslate_step = max(steps, round(xTranslate_step / 2));
		yTranslate_step = max(steps, round(yTranslate_step / 2));
		rotate_step = max(steps, round(rotate_step / 2));
		scale_step = max(steps, round(scale_step / 2));
	}
	if (sccenter + scrange / 2 > 1) {
		sccenter = 1.0 - scrange / 2;
	}

	if (xTranslate_layer == true) {
		double xTranslate1 = xTcenter;
		double xTranslate2 = xTcenter;
		for (int i = 0; i <= ceil(xTranslate_step / 2); i++) {
			affine_transformation = (Mat_<float>(1, 6) << 1, 0, xTranslate1, 0, 1, 0);
			transformations.push_back(affine_transformation);
			xT_val.push_back(xTranslate1);
			xTranslate1 += xTrange / xTranslate_step;
			if (i != 0) {
				affine_transformation = (Mat_<float>(1, 6) << 1, 0, xTranslate2, 0, 1, 0);
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
			affine_transformation = (Mat_<float>(1, 6) << 1, 0, 0, 0, 1, yTranslate1);
			transformations.push_back(affine_transformation);
			yT_val.push_back(yTranslate1);
			yTranslate1 += yTrange / yTranslate_step;
			if (i != 0) {
				affine_transformation = (Mat_<float>(1, 6) << 1, 0, 0, 0, 1, yTranslate2);
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
			affine_transformation = (Mat_<float>(1, 6) << cos(thetad), -sin(thetad), 0, sin(thetad), cos(thetad), 0);
			transformations.push_back(affine_transformation);
			rot_val.push_back(theta1);
			if (i != 0 && i != ceil(rotate_step / 2)) {
				thetad = theta2 * PI / 180;
				affine_transformation = (Mat_<float>(1, 6) << cos(thetad), -sin(thetad), 0, sin(thetad), cos(thetad), 0);
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
				affine_transformation = (Mat_<float>(1, 6) << scale, 0, 0, 0, scale, 0);
				transformations.push_back(affine_transformation);
				sc_val.push_back(scale);
				scale -= (maxscale_para - minscale_para) / scale_step;
			}
		}
		else {
			double sc1 = sccenter;
			double sc2 = sccenter;
			for (int i = 0; i <= ceil(scale_step / 2); i++) {
				affine_transformation = (Mat_<float>(1, 6) << sc1, 0, 0, 0, sc1, 0);
				transformations.push_back(affine_transformation);
				sc_val.push_back(sc1);
				sc1 += scrange / scale_step;
				if (i != 0) {
					affine_transformation = (Mat_<float>(1, 6) << sc2, 0, 0, 0, sc2, 0);
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

void getTransMap(Size img_size, int flag, vector<Mat> &ResultMap) {
	int count = 0;
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
	int height = img_size.height;
	int width = img_size.width;

	if (xTranslate_layer) {
		Mat Map(xT_val.size(), pixcount, CV_32SC1);
		for (int k = 0; k < xT_val.size(); k++) {
			/* for all the xtranslations*/
			pos = 0;
			Mat temp(height, width, CV_32SC1);
			int* pt = (int*)temp.data;
			int xT = round(xT_val[k])*flag;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (j - xT<0 || j - xT >= width) {
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
			Mat temp(height, width, CV_32SC1);
			int* pt = (int*)temp.data;
			int yT = round(yT_val[k])*flag;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (i < yT || i - yT >= height) {
						pt[pos] = -1;
					}
					else {
						pt[pos] = pos - width*yT;
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
			Mat temp(height, width, CV_32SC1);
			int* pt = (int*)temp.data;
			double ang = rot_val[k] * flag*PI / 180;
			int x, y, newx, newy;
			double cosang, sinang;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					x = j - width / 2;
					y = i - height / 2;
					cosang = cos(ang);
					sinang = sin(ang);
					newx = cosang*x - sinang*y;
					newy = sinang*x + cosang*y;
					if (abs(newx) >= width / 2 || abs(newy) >= height / 2)
						pt[pos] = -1;
					else
						pt[pos] = floor(newx + width / 2.0) + floor(newy + height / 2.0)*width;
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
			Mat temp(height, width, CV_32SC1);
			int* pt = (int*)temp.data;
			double sc = flag == 1 ? sc_val[k] : 1 / sc_val[k];
			int x, y, newx, newy, cosang, sinang;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					x = j - width / 2;
					y = i - height / 2;
					newx = sc*x;
					newy = sc*y;
					if (abs(newx) >= width / 2 || abs(newy) >= height / 2)
						pt[pos] = -1;
					else
						pt[pos] = floor(newx + width / 2.0) + floor(newy + height / 2.0)*width;
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


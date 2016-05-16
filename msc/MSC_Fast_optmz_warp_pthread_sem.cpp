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
#include <semaphore.h>

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
double minscale_para = 0.5;	/*minimum scaling factor*/

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
Fwd_Path_Values *FPV;
Mat *BPV;
sem_t* sem_gout;
sem_t* sem_gin;
sem_t* sem_fwd;
sem_t* sem_bwd;
pthread_mutex_t* mutex_g;
int* idxTrans;
int mscFlag = 1;

/* scaling control*/
/* start scaling normalizer after all the rest layer are clear*/
int startscale = 1;

uint32_t t_total, t_loop;
vector< Mat > G;
Size img_size;
int iteration_count = 100; // Number of iterations for which MSC will operate.
int count = 0;

void getTransform(Size img_size, vector < Mat > &transformation_set, vector< Mat > &G, TransformationSet lastTr = TransformationSet());

/* get the mapping of pixels for all the forward transformation*/
void getTransMap(Size img_size, int flag, vector<Mat> &ResultMap);

int SL_MSC(Mat Input_Image, Mat Memory_Images, Size input_size, Mat *Fwd_Path, Mat *Bwd_Path, TransformationSet & finalTrans){
	t_total = clock();
	t_loop = 0;

	Mat affine_transformation;
	double MAX_VAL = 255;
	Mat transformations;
	vector < Mat > transformation_set;
	img_size = input_size;

    //vector< Mat > G; // The competition function
    int ret = -1;
    double verified_ret;
    FILE *fp;
    double dot_product_input_object = Input_Image.dot(Input_Image);

    
    layer_count = 1+(int)(xTranslate_layer)+(int)(yTranslate_layer)+(int)(rotate_layer)+(int)(scale_layer);
     
    //transformations.release();
    //G_layer.release();
    
    
    k_transformations = new double[layer_count-1];
    
   // k_transformations[layer_count-1] = k_memory;

		getTransform(img_size, transformation_set, G, finalTrans);

	//transformation_set.clear();

	//vector<Mat>().swap(transformation_set);

	/* get the mapping for transformation*/
	//vector<Mat> MapForw, MapBack;
	//MapForw.reserve(layer_count - 1);
	//MapBack.reserve(layer_count - 1);
	//getTransMap(img_size, FORWARD, MapForw);
	//getTransMap(img_size, BACKWARD, MapBack);



	idxTrans = new int[layer_count - 1];

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

	int status;
	sem_gout = new sem_t[layer_count - 1];
	sem_gin = new sem_t[layer_count - 1];
	sem_fwd = new sem_t[layer_count - 1];
	sem_bwd = new sem_t[layer_count - 1];
	mutex_g = new pthread_mutex_t[layer_count - 1];

	for (int i = 1; i < layer_count; i++) {
		status = sem_init(&sem_gout[i-1], 0, 2);
		if (status != 0) { perror("sem_init failed"); exit(status); }
		status = sem_init(&sem_gin[i - 1], 0, 0);
		if (status != 0) { perror("sem_init failed"); exit(status); }
		status = sem_init(&sem_fwd[i-1], 0, i==1?1:0);
		if (status != 0) { perror("sem_init failed"); exit(status); }
		status = sem_init(&sem_bwd[layer_count-1-i], 0, i == 1 ? 1 : 0);
		if (status != 0) { perror("sem_init failed"); exit(status); }
		//status = pthread_mutex_init(&mutex_g[i-1], NULL);
		//if (status != 0) { perror("mutex_init failed"); exit(status); }
	}
	
	pthread_t* threadFwd = new pthread_t[layer_count - 1];
	pthread_t* threadBwd = new pthread_t[layer_count - 1];
	pthread_t* threadUpd = new pthread_t[layer_count - 1];

	int layers = layer_count;
	FwtArg *fwrd = new FwtArg[layers - 1];
	BktArg *bckwrd = new BktArg[layers - 1];
	UpdArg *update = new UpdArg[layers - 1];
	Mat *TranSc = new Mat_<float>[layers];
	for (int i = 1; i < layers; i++) {
		// Perform all of the forward path transformations
		TranSc[i - 1] = Mat(Size(G[i - 1].cols, 1), CV_32F);


		fwrd[i-1].pg = &G[i - 1];
		fwrd[i - 1].pIn = &FPV[i - 1].Fwd_Superposition;
		fwrd[i - 1].pInFP = &FPV[i];
		fwrd[i - 1].pTransc = &TranSc[i - 1];
		fwrd[i - 1].ptrans = &(transformation_set[i - 1]);
		fwrd[i - 1].ret = &FPV[i];
		fwrd[i - 1].nlayer = i-1;
		//pthread_t tid1;
		void *status1;
		pthread_create(&threadFwd[i-1], NULL, ForwardTransform, &fwrd[i - 1]);
		//ForwardTransform(&fwrd);


		bckwrd[i-1].pg = &G[layers - 1 - i];
		bckwrd[i - 1].pIn = &BPV[layers - i];
		bckwrd[i - 1].ptrans = &(transformation_set[layers - i - 1]);
		bckwrd[i - 1].ret = &BPV[layers - 1 - i];
		bckwrd[i - 1].nlayer = layers - 1 - i;
		//pthread_t tid2;
		void *status2;
		pthread_create(&threadBwd[layers-i-1],NULL, BackwardTransform, &bckwrd[i - 1]);
		//BackwardTransform(&bckwrd);


		update[i - 1].nlayer = i;
		update[i - 1].pfinalTrans = &finalTrans;
		update[i - 1].pg = &G[i-1];
		update[i - 1].pTranSc = &TranSc[i - 1];
		pthread_create(&threadUpd[i-1], NULL, UpdateCompetition, &update[i - 1]);
	}

	for (int i = 1; i < layers; i++) {
		void* status;
		pthread_join(threadFwd[i - 1], &status);
		pthread_join(threadBwd[i - 1], &status);
		pthread_join(threadUpd[i - 1], &status);
	}
  
	for (int i = 1; i < layer_count; i++) {
		status = sem_destroy(&sem_gout[i - 1]);
		if (status != 0) { perror("sem_destroy failed"); exit(status); }
		status = sem_destroy(&sem_gin[i - 1]);
		if (status != 0) { perror("sem_destroy failed"); exit(status); }
		status = sem_destroy(&sem_fwd[i - 1]);
		if (status != 0) { perror("sem_destroy failed"); exit(status); }
		status = sem_destroy(&sem_bwd[layer_count - 1 - i]);
		if (status != 0) { perror("sem_destroy failed"); exit(status); }
		//status = pthread_mutex_init(&mutex_g[i-1], NULL);
		//if (status != 0) { perror("mutex_init failed"); exit(status); }
	}

	ret = MapSeekingCircuit(Input_Image, Memory_Images, img_size, Fwd_Path, Bwd_Path, layer_count, transformation_set, &G, k_transformations);
	//printf("The value of verified_ret is %g\n", verified_ret);

	//xT_val.clear();
	//yT_val.clear();
	//rot_val.clear();
	//sc_val.clear();

	vector<double>().swap(xT_val);
	vector<double>().swap(yT_val);
	vector<double>().swap(rot_val);
	vector<double>().swap(sc_val);

	//MapForw.clear();
	//vector<Mat>().swap(MapForw);
	//MapBack.clear();
	//G.clear();
	//vector<Mat>().swap(MapBack);
	vector<Mat>().swap(G);

	t_total = clock()-t_total;

	printf("it takes %d/%d for loop\n", t_loop, t_total);
    return ret;
    
}

int MapSeekingCircuit(Mat Input_Image, Mat Memory_Images, Size img_size, Mat *Fwd_Path, Mat *Bwd_Path, int layers, vector< Mat > image_transformations, vector< Mat > *G, double k_transformations[]){
    
    vector< Mat > g = *G;
    
    //Fwd_Path_Values *FPV = new Fwd_Path_Values[layers];
    
    //Mat *BPV = new Mat[layers];

	Mat *TranSc = new Mat_<float>[layers];
    
    /*
     The transformation matrix in openCV looks like:
     
     [                       |                   |
           xscale*cos(theta) |    -sin(theta)    |  x-translation
            sin(theta)       |  yscale*cos(theta)|  y-translation
                 0           |        0          |    1
     ]
     */
    
    //FPV[0].Fwd_Superposition = Input_Image.clone();
	//FPV[1].Transformed_Templates = Mat::zeros(Size(Input_Image.rows*Input_Image.cols, g[0].cols), CV_32F);
    
    //printf("Backward path superposition \n");
   // BPV[layers-1] = Superimpose_Memory_Images(Memory_Images, g[layers-1], img_size.height).clone();

	threshold(Memory_Images, BPV[layers - 1], 1, MAX_VAL, THRESH_TRUNC);
	BPV[layers - 1].convertTo(BPV[layers - 1], CV_32FC1);
    
    
    //imshow("BPV[layers-1]", BPV[layers-1]*255);
    if(layers>1){
        //printf("Apply transformations\n");
        for(int i = 1; i < layers; i++){
            // Perform all of the forward path transformations
			TranSc[i - 1] = Mat(Size(g[i-1].cols, 1), CV_32F);

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

			BktArg bckwrd;
			bckwrd.pg = &g[layers - 1 - i];
			bckwrd.pIn = &BPV[layers - i];
			bckwrd.ptrans = &(image_transformations[layers - i - 1]);
			bckwrd.ret = &BPV[layers - 1 - i];
			pthread_t tid2;
			void *status2;
			//pthread_create(&tid2, NULL, BackwardTransform, &bckwrd);
			BackwardTransform(&bckwrd);

			//pthread_join(tid1, &status1);
			//pthread_join(tid2, &status2);
            //FPV[i] = ForwardTransform(FPV[i-1].Fwd_Superposition, FPV[i], image_transformations[i - 1].clone(), g[i-1],TranSc[i-1]);
			//cout << TranSc[i - 1];
            // Perform all of the backward path transformations
            //BPV[layers-1-i] = BackwardTransform(BPV[layers-i], image_transformations[layers - i - 1].clone(), g[layers-1-i]);
        }
        
        //printf("Update competition function\n");
//#pragma omp parallel for
        for(int i = 1; i < layers; i++){
            // Update competition
            g[i-1] = UpdateCompetition(FPV[i].Transformed_Templates, BPV[i], g[i-1], img_size.height, k_transformations[i-1], TranSc[i - 1]).clone();
            
			if (dispg)
				cout<<"g"<<i-1<<"="<<g[i-1]<<endl;
            //cout<<endl;
        }
        
    }

    //
 //   imshow("FPV_Forward[1]", (FPV[layers-1].Fwd_Superposition)*255);
 //   
 //   //imshow("BPV[1]", (BPV[0])*255);
	//waitKey();
 //   
    *Fwd_Path = FPV[layers-1].Fwd_Superposition.clone();
    *Bwd_Path = BPV[0].clone();
    
    *G = g;

	//delete[] FPV;

	for (int i = 0; i < layers; i++) {
		//BPV[i].release();
		TranSc[i].release();
	}

	
    return 0;
}

void *ForwardTransform(void* pargin) {
	FwtArg* argin = (FwtArg*)pargin;
	Mat In = *argin->pIn;
	Fwd_Path_Values* InFP = argin->pInFP;
	Mat Perspective_Transformation_Matrix = (*argin->ptrans).clone();
	Mat g = (*argin->pg).clone();
	Mat Transc = *argin->pTransc;
	int nlayer = argin->nlayer;
	Fwd_Path_Values* FPV_return = argin->ret;
	Mat SuperPosition;
	Mat TransformedTemplates;
	double Thresh_VAL = 100;
	float sine;
	float cosine;
	float angle;
	double matrix_determinant;
	Mat rotation_matrix;
	int count = g.cols;

	Mat temp;
	Mat& retTemp = InFP->Transformed_Templates;

	int status;

	while (mscFlag) {
		status = sem_wait(&sem_gout[nlayer]);
		if (status != 0) {
			perror("sem wait failed"); exit(status);
		}
		if (nlayer != 0) {
			status = sem_wait(&sem_fwd[nlayer - 1]);
			if (status != 0) {
				perror("sem wait failed"); exit(status);
			}
		}


		//dst = In.clone();
		SuperPosition = (InFP->Fwd_Superposition).setTo(0);



		for (int i = 0; i<count; i++) {
			Mat dst(In.rows, In.cols, CV_32F);
			if (g.at<float>(0, i) == 0) {
				//Mat temp1 = Mat::zeros(Size((In.rows)*(In.cols),1), CV_32F);
				//retTemp.push_back(temp1);
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

				//imshow("In", In);
				//imshow("dst", dst);
				//cvWaitKey();
			}
			else {
				warpAffine(In, dst, Perspective_Transformation_Matrix_2D, dst.size(), INTER_NEAREST);
				Transc.at<float>(0, i) = (1.0);
			}
			t_loop += clock() - t_temp;


			if (0) {
				dst.convertTo(temp, CV_8U, 255);
				imshow("temp", temp); waitKey();
				temp = CannyThreshold(temp, 50, 100);
				imshow("tempafter", temp); waitKey();
				temp.clone().convertTo(dst, CV_32F, 1.0 / 255);
			}
			if (test_forw) {
				imshow("In1", g.at<float>(0, i)*In * 255);
				imshow("dst1", g.at<float>(0, i)*dst * 255);
				cvWaitKey();
			}
			dst.convertTo(dst, CV_32FC1);
			Mat dst_scaled = g.at<float>(0, i)*dst;
			SuperPosition = SuperPosition + dst_scaled;

			dst_scaled.reshape(0, 1).copyTo(retTemp.row(i));


			dst.release();
		}
		FPV_return->Transformed_Templates = retTemp;
		threshold(SuperPosition, SuperPosition, Thresh_VAL, MAX_VAL, THRESH_TRUNC);
		FPV_return->Fwd_Superposition = SuperPosition.clone();
		
		if (nlayer!=layer_count-2)	sem_post(&sem_fwd[nlayer+1]);
		sem_post(&sem_gin[nlayer]);
	}
	return 0;
}

void *BackwardTransform(void* pArgin) {
	BktArg* argin = (BktArg*)pArgin;
	Mat In = *argin->pIn;
	Mat Perspective_Transformation_Matrix = (*argin->ptrans).clone();
	Mat g = (*argin->pg).clone();
	Mat* BPV_return = argin->ret;
	Mat& SuperPosition = *argin->ret;
	int nlayer = argin->nlayer;

	Mat TransformedTemplates;
	float sine;
	double Thresh_VAL = 1;
	float cosine;
	float angle;
	double matrix_determinant;
	Mat rotation_matrix;
	int count = g.cols;
	//g.convertTo(g, CV_32F);
	Mat dst(In.rows, In.cols, CV_32F);

	int status;
	while(mscFlag) {
		status = sem_wait(&sem_gout[nlayer]);
		if (status != 0) {
			perror("sem wait failed"); exit(status);
		}
		if (nlayer != layer_count-2) {
			status = sem_wait(&sem_bwd[nlayer + 1]);
			if (status != 0) {
				perror("sem wait failed"); exit(status);
			}
		}

		SuperPosition.setTo(0);

		for (int i = 0; i<count; i++) {

			if (g.at<float>(0, i) == 0) {
				continue;
			}

			uint32_t t_temp = clock();
			Mat Perspective_Transformation_Matrix_2D = Perspective_Transformation_Matrix.row(i).reshape(0, 2);

			sine = -Perspective_Transformation_Matrix_2D.at<float>(0, 1);
			cosine = Perspective_Transformation_Matrix_2D.at<float>(0, 0);

			angle = (-1)*roundf(atan(sine / cosine) * 180 / PI);

			matrix_determinant = (sqrt(Perspective_Transformation_Matrix_2D.at<float>(0, 0)*Perspective_Transformation_Matrix_2D.at<float>(1, 1) - Perspective_Transformation_Matrix_2D.at<float>(0, 1)*Perspective_Transformation_Matrix_2D.at<float>(1, 0)));
			if (abs(angle) >= 0.0001 || abs(matrix_determinant - 1) >= 0.0001) {
				Point2f src_center(In.cols / 2.0F, In.rows / 2.0F);
				rotation_matrix = getRotationMatrix2D(src_center, angle, matrix_determinant);
				warpAffine(In, dst, rotation_matrix, dst.size(), INTER_NEAREST);
			}
			else {
				Perspective_Transformation_Matrix_2D.at<float>(0, 2) = (-1)*Perspective_Transformation_Matrix_2D.at<float>(0, 2);
				Perspective_Transformation_Matrix_2D.at<float>(1, 2) = (-1)*Perspective_Transformation_Matrix_2D.at<float>(1, 2);
				warpAffine(In, dst, Perspective_Transformation_Matrix_2D, dst.size(), INTER_NEAREST);
			}

			t_loop += clock() - t_temp;

			if (test_back) {
				imshow("In", g.at<float>(0, i)*In * 255);
				imshow("dst", g.at<float>(0, i)*dst * 255);
				cvWaitKey();
			}

			dst.convertTo(dst, CV_32FC1);

			Mat dst_scaled = g.at<float>(0, i)*dst;
			SuperPosition = SuperPosition + dst_scaled;

			dst.release();
		}
		threshold(SuperPosition, SuperPosition, Thresh_VAL, MAX_VAL, THRESH_TRUNC);

		if (nlayer!=0)	sem_post(&sem_bwd[nlayer-1]);
		sem_post(&sem_gin[nlayer]);
	}

	
	//Mat Superposition_image_changed = foregroundBackgroundImageChange(SuperPosition);
	//BPV_return = &(SuperPosition);
	//SuperPosition.copyTo(*BPV_return);
	//BPV_return = &SuperPosition.clone();

	return 0;
}


void * UpdateCompetition(void* pArgin) {
	UpdArg* argin = (UpdArg*)pArgin;
	int nlayer = argin->nlayer;
	Mat Transformed_Templates = FPV[nlayer].Transformed_Templates;
	Mat BackwardTransform = BPV[nlayer];
	Mat& g = *argin->pg; 
	Mat TranSc = *argin->pTranSc;
	double k = k_transformations[nlayer-1];
	double p = 1;
	int r = img_size.height;
	TransformationSet& finalTrans = *argin->pfinalTrans;

	int count = Transformed_Templates.rows;
	//Mat subtracted_g(g.rows, g.cols, CV_64FC1);
	//Mat thresholded_g(g.rows, g.cols, CV_32FC1);
	double Thresh_VAL = 0.1;
	double MAX_VAL = 1;
	Mat q(g.rows, g.cols, CV_32F);
	double T_L2;
	double BackwardTransform_L2;
	double min, max;

	int status;
	while (mscFlag) {
		status = sem_wait(&sem_gin[nlayer-1]);
		if (status != 0) {
			perror("sem wait failed"); exit(status);
		}
		status = sem_wait(&sem_gin[nlayer-1]);

		if (status != 0) {
			perror("sem wait failed"); exit(status);
		}
		for (int i = 0; i<count; i++) {
			if (g.at<float>(0, i) == 0) {
				q.at<float>(0, i) = 0;
				continue;
			}
			Mat T = Transformed_Templates.row(i).reshape(0, r);
			T.convertTo(T, CV_32FC1);
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
			if (BackwardTransform_L2 != 0 && T_L2 != 0) {
				if (startscale)
					/*q.at<float>(0,i) = T.dot(BackwardTransform)*(pow(TranSc.at<float>(0,i),0.8));*/
					q.at<float>(0, i) = T.dot(BackwardTransform)*(TranSc.at<float>(0, i));
				//q.at<double>(0, i) = T.dot(BackwardTransform) / T_L2;
				else
					q.at<float>(0, i) = T.dot(BackwardTransform);
			}
			else {
				q.at<float>(0, i) = 0;
			}
			if (startscale)
				if (TranSc.at<float>(0, i) < 1)
					p = 1;

		}

		//cout<<"q: "<<q<<endl;
		minMaxLoc(q, &min, &max);
		// cout<<"q_min:"<<min<<"  q_max: "<<max<<endl;
		Mat temp;
		pow(1 - q / max, p, temp);
		subtract(g, k*(temp), g);
		//cout << "g:" << g << "  subtracted_g: " << subtracted_g << endl;
		g.convertTo(g, CV_32F);
		threshold(g, g, Thresh_VAL, MAX_VAL, THRESH_TOZERO);

		sem_post(&sem_gout[nlayer-1]);
		sem_post(&sem_gout[nlayer-1]);

		if (nlayer == layer_count - 2) {
			iteration_count--;
			if (iteration_count % 5 == 0) {
				///* only inspect before the scaling layer*/

				for (int kk = 0; kk < G.size(); kk++) {
					//cout << "-------\n" << G[kk] << "-------\n";
					if (countNonZero(G[kk]) != 1) {
						mscFlag = 1;
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

				/* stop iteration condition: only one transformation is left*/
				/* record the final transformation*/
				if (!mscFlag) {
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
		}
		
	}
	return 0;

}

Mat UpdateCompetition(Mat Transformed_Templates, Mat BackwardTransform, Mat g, int r, double k, Mat TranSc,double p){
    int count = Transformed_Templates.rows;
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


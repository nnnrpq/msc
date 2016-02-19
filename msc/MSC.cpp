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

Mat affine_transformation;
double MAX_VAL = 255;
Mat transformations;
vector < Mat > transformation_set;


bool xTranslate_layer = true;
bool yTranslate_layer = true;
bool rotate_layer = false;
bool scale_layer = true;

double k_xTranslate = 0.5;
double k_yTranslate = 0.5;
double k_rotate = 0.45;
double k_scale = 0.35;

double k_memory = 0.25;



double *k_transformations;

Mat C = (Mat_<double>(1,3) << 0, 0, 1);


int SL_MSC(Mat Input_Image, Mat Memory_Images, vector<int> row_size, Mat *Fwd_Path, Mat *Bwd_Path){
    
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
    for(int i=1; i<layer_count; i++){
        char Layer_filename[200];
        char layer_name[40];
        printf("iteration #: %d\n", i);
        strcpy(Layer_filename, "Layer_");
        
        if(xTranslate_layer==true){
            strcpy(layer_name,"xTranslate_layer");
            xTranslate_layer=false;
            k_transformations[i-1] = k_xTranslate;
        }else if(yTranslate_layer==true){
            strcpy(layer_name,"yTranslate_layer");
            yTranslate_layer=false;
            k_transformations[i-1] = k_yTranslate;
        }else if(rotate_layer==true){
            strcpy(layer_name,"rotate_layer");
            rotate_layer=false;
            k_transformations[i-1] = k_rotate;
        }else if(scale_layer==true){
            strcpy(layer_name,"scale_layer");
            k_transformations[i-1] = k_scale;
        }
        
        strcat(Layer_filename, layer_name);
        strcat(Layer_filename, ".txt");
        fp = fopen(Layer_filename, "r+");
        float next = 0;
        int count = 0;
        while( fscanf(fp, "%f ", &next) > 0 ) // parse %f followed by ' '
        {
            affine_transformation.push_back(next);
            count++;
            if(count == 9){
                transformations.push_back(affine_transformation.reshape(0,1));
                G_layer.push_back(1);
                affine_transformation.release();
                count = 0;
            }
        }
        
        fclose(fp);
        transformation_set.push_back(transformations);
        if(i<layer_count){
            G.push_back(G_layer.reshape(0,1));
        }
        //cout<<"Transformations"<<endl;
        //cout<<transformations<<endl;
        G_layer.release();
        transformations.release();
    }
    
    for(int i = 0; i < Memory_Images.rows; i++){
        G_layer.push_back(1);
    }
    G.push_back(G_layer.reshape(0,1));
    G_layer.release();
    
    
    while(iteration_count > 0){
        iteration_count--;
        printf("About to call MSC\n");
        ret = MapSeekingCircuit(Input_Image, Memory_Images, row_size, Fwd_Path, Bwd_Path, layer_count, transformation_set, &G, k_transformations);
        
        if(iteration_count %10 == 0){
            cvWaitKey(0);
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
    printf("The value of verified_ret is %g\n", verified_ret);
    return ret;
    
}

int MapSeekingCircuit(Mat Input_Image, Mat Memory_Images, vector<int> row_size, Mat *Fwd_Path, Mat *Bwd_Path, int layers, vector< Mat > image_transformations, vector< Mat > *G, double k_transformations[]){
    
    vector< Mat > g = *G;
    
    Fwd_Path_Values *FPV = new Fwd_Path_Values[layers];
    
    Mat *BPV = new Mat[layers];
    
    /*
     The transformation matrix in openCV looks like:
     
     [                       |                   |
           xscale*cos(theta) |    -sin(theta)    |  x-translation
            sin(theta)       |  yscale*cos(theta)|  y-translation
                 0           |        0          |    1
     ]
     */
    
    FPV[0].Fwd_Superposition = Input_Image.clone();
    
    printf("Backward path superposition \n");
    BPV[layers-1] = Superimpose_Memory_Images(Memory_Images, g[layers-1], row_size[0]).clone();
    
    
    
    //imshow("BPV[layers-1]", BPV[layers-1]*255);
    if(layers>1){
        printf("Apply transformations\n");
        for(int i = 1; i < layers; i++){
            // Perform all of the forward path transformations
            FPV[i] = ForwardTransform(FPV[i-1].Fwd_Superposition, image_transformations[i-1], g[i-1]);
            
            // Perform all of the backward path transformations
            BPV[layers-1-i] = BackwardTransform(BPV[layers-i], image_transformations[layers-i-1], g[layers-1-i]);
        }
        
        //printf("Update competition function\n");
        for(int i = 1; i < layers; i++){
            // Update competition
            g[i-1] = UpdateCompetition(FPV[i].Transformed_Templates, BPV[i], g[i-1], row_size[0], k_transformations[i-1]).clone();
            
            //cout<<g[i-1]<<endl;
            //cout<<endl;
        }
        
    }
    //cout<<"layers-1  "<<layers-1<<endl;
    g[layers-1] = UpdateCompetition_Memory(Memory_Images, FPV[layers-1].Fwd_Superposition, g[layers-1], row_size[0], k_transformations[layers-1]).clone();
    
    //cout<<"g memory: "<<g[layers-1]<<endl;
    
    imshow("FPV_Forward[1]", (FPV[layers-1].Fwd_Superposition)*255);
    
    imshow("BPV[1]", (BPV[0])*255);
    
    *Fwd_Path = FPV[layers-1].Fwd_Superposition;
    *Bwd_Path = BPV[0];
    
    *G = g;
    return 0;
}

Fwd_Path_Values ForwardTransform(Mat In, Mat Perspective_Transformation_Matrix, Mat g){
    Fwd_Path_Values FPV_return;
    Mat SuperPosition;
    Mat TransformedTemplates;
    double Thresh_VAL = 1;
    float sine;
    float cosine;
    float angle;
    double matrix_determinant;
    Mat rotation_matrix;
    int count = g.cols;
    g.convertTo(g,CV_64F);
    Mat dst;
    dst = Mat::zeros(In.rows, In.cols, CV_32FC1);
    SuperPosition = Mat::zeros(In.rows, In.cols, CV_32FC1);
    SuperPosition = g.at<double>(0,0)*In.clone();
    FPV_return.Transformed_Templates.push_back(SuperPosition.reshape(0,1));
    for(int i=1; i<count; i++){
        Mat Perspective_Transformation_Matrix_2D = Perspective_Transformation_Matrix.row(i).reshape(0,3);
        
        sine = Perspective_Transformation_Matrix_2D.at<float>(0,1);
        cosine = Perspective_Transformation_Matrix_2D.at<float>(0,0);
        
        angle = roundf(atan(sine/cosine) * 180/PI);
        
        matrix_determinant = (sqrt(determinant(Perspective_Transformation_Matrix_2D)));
	//cout<<"Perspective Matrix Forward: "<<Perspective_Transformation_Matrix_2D<<endl;
        //cout<<"Matrix D Forward   "<<matrix_determinant<<endl;
        if(angle !=0 || matrix_determinant != 1.0){
            //cout<<"Perspective Matrix Forward: "<<Perspective_Transformation_Matrix_2D<<endl;
            Point2f src_center(In.cols/2.0F, In.rows/2.0F);
            rotation_matrix = getRotationMatrix2D(src_center, angle, matrix_determinant);
            vconcat(rotation_matrix, C, rotation_matrix);
            warpPerspective( In, dst, rotation_matrix, dst.size() );
        }else{
            warpPerspective( In, dst, Perspective_Transformation_Matrix_2D, dst.size() );
        }
        /*
        if(g.at<double>(0,i) < 0.3){
            g.at<double>(0,i) = 0;
        }
         */
        Mat dst_scaled = g.at<double>(0,i)*dst;
        SuperPosition = SuperPosition + dst_scaled;
        FPV_return.Transformed_Templates.push_back(dst_scaled.reshape(0,1));
        dst.release();
    }
    //SuperPosition.convertTo(SuperPosition,CV_8U);
    threshold(SuperPosition, SuperPosition, Thresh_VAL, MAX_VAL, THRESH_TRUNC);
    FPV_return.Fwd_Superposition = SuperPosition.clone();
    return FPV_return;
}


Mat BackwardTransform(Mat In, Mat Perspective_Transformation_Matrix, Mat g){
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
    g.convertTo(g,CV_64F);
    Mat dst;
    dst = Mat::zeros(In.rows, In.cols, CV_32FC1);
    SuperPosition = Mat::zeros(In.rows, In.cols, CV_32FC1);
    SuperPosition = g.at<double>(0,0)*In.clone();
    
    for(int i=1; i<count; i++){
        Mat Perspective_Transformation_Matrix_2D = Perspective_Transformation_Matrix.row(i).reshape(0,3);
        
        sine = Perspective_Transformation_Matrix_2D.at<float>(0,1);
        cosine = Perspective_Transformation_Matrix_2D.at<float>(0,0);
        
        angle = (-1)*roundf(atan(sine/cosine) * 180/PI);
        
        matrix_determinant = (1/(sqrt(determinant(Perspective_Transformation_Matrix_2D))));
        //cout<<"Perspective matrix backward :"<<Perspective_Transformation_Matrix_2D<<endl;
        //cout<<"Matrix D Backward   "<<matrix_determinant<<endl;
        if(angle !=0 || matrix_determinant != 1.0){
            //cout<<"Perspective matrix backward :"<<Perspective_Transformation_Matrix_2D<<endl;
            Point2f src_center(In.cols/2.0F, In.rows/2.0F);
            rotation_matrix = getRotationMatrix2D(src_center, angle, matrix_determinant);
            vconcat(rotation_matrix, C, rotation_matrix);
            warpPerspective( In, dst, rotation_matrix, dst.size() );
        }else{
            Perspective_Transformation_Matrix_2D.at<float>(0,2) = (-1)*Perspective_Transformation_Matrix_2D.at<float>(0,2);
            Perspective_Transformation_Matrix_2D.at<float>(1,2) = (-1)*Perspective_Transformation_Matrix_2D.at<float>(1,2);
            warpPerspective( In, dst, Perspective_Transformation_Matrix_2D, dst.size() );
        }
        
        Mat dst_scaled = g.at<double>(0,i)*dst;
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
        Superimposed_Image = (Superimposed_Image + g.at<double>(0,i)*M.row(i));
    }
    Superimposed_Image.convertTo(Superimposed_Image,CV_32FC1);
    //printf("Memory superposition\n");
    threshold(Superimposed_Image, Superimposed_Image, Thresh_VAL, MAX_VAL, THRESH_TRUNC);
    return Superimposed_Image.reshape(0,r);
}

Mat UpdateCompetition(Mat Transformed_Templates, Mat BackwardTransform, Mat g, int r, double k){
    int count = Transformed_Templates.rows;
    g.convertTo(g,CV_64F);
    Mat subtracted_g(g.rows, g.cols, CV_64FC1);
    Mat thresholded_g(g.rows, g.cols, CV_32FC1);
    double Thresh_VAL = 0.3;
    double MAX_VAL = 1;
    Mat q(g.rows, g.cols, CV_64F);
    double T_L2;
    double BackwardTransform_L2;
    double min, max;
    for(int i=0; i<count; i++){
        Mat T = Transformed_Templates.row(i).reshape(0,r);
        T.convertTo(T,CV_32FC1);
        T_L2 = norm(T, NORM_L2);
        BackwardTransform_L2 = norm(BackwardTransform, NORM_L2);
        
        
        if(BackwardTransform_L2 !=0 && T_L2 != 0){
            q.at<double>(0,i) = T.dot(BackwardTransform)/(T_L2*BackwardTransform_L2);
        }else{
            q.at<double>(0,i) = 0;
        }
    }
    cout<<"q: "<<q<<endl;
    minMaxLoc(q, &min, &max);
    //cout<<"q_min:"<<min<<"  q_max: "<<max<<endl;
    subtract(g, k*(1-q/max), subtracted_g);
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
    cout<<"q: "<<q<<endl;
    subtracted_g.convertTo(subtracted_g,CV_32FC1);
    threshold(subtracted_g, thresholded_g, Thresh_VAL, MAX_VAL, THRESH_TOZERO);
    return thresholded_g;
}
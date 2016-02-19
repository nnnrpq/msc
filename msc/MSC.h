#ifndef SLMSC_H_INCLUDED
#define SLMSC_H_INCLUDED
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include<stdio.h>
#include <vector>
using namespace cv;
using namespace std;
class Fwd_Path_Values{
public:
    Mat Fwd_Superposition;
    Mat Transformed_Templates;
};

int SL_MSC(Mat , Mat , vector<int> , Mat *, Mat *);
int MapSeekingCircuit(Mat , Mat , vector<int> , Mat *, Mat *, int, vector< Mat > , vector< Mat > *, double []);
int Verify_Image(Mat , Mat );
Fwd_Path_Values ForwardTransform(Mat , Mat ,Mat  );
Mat BackwardTransform(Mat , Mat , Mat );
Mat Superimpose_Memory_Images(Mat , Mat, int );
Mat UpdateCompetition(Mat, Mat, Mat ,int , double);
Mat UpdateCompetition_Memory(Mat, Mat, Mat ,int , double);

Mat UpdateCompetition_Background(Mat, Mat, Mat, Mat, Mat ,int );
Mat UpdateCompetition_Memory_Background(Mat, Mat, Mat, Mat, Mat ,int );

Fwd_Path_Values ForwardTransform_Background(Mat , Mat ,Mat  );
Mat BackwardTransform_Background(Mat , Mat , Mat );

#endif
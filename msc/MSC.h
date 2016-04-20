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
	~Fwd_Path_Values() {
		Fwd_Superposition.release();
		Transformed_Templates.release();
	}
};
class TransformationSet {
public:
	int nonIdenticalCount;		/* show what is in the transformation*/
	double xTranslate;
	double yTranslate;
	double theta;
	double scale;
	TransformationSet(double xT = 0, double yT = 0, double ang = 0, double sc = 0) {
		xTranslate = xT;
		yTranslate = yT;
		theta = ang;
		scale = sc;
		nonIdenticalCount = (xT != 0) + (yT != 0) + (ang != 0) + (sc != 0);
	}
};

int SL_MSC(Mat , Mat , Size , Mat *, Mat *,TransformationSet &);

int MapSeekingCircuit(Mat , Mat , Size , Mat *, Mat *, int, vector< Mat > , vector< Mat > *, double []);
int MapSeekingCircuit(Mat, Mat, Size, Mat *, Mat *, int, vector< Mat >, vector< Mat >, vector< Mat > *, double[]);

int Verify_Image(Mat , Mat );
Fwd_Path_Values ForwardTransform(Mat , Mat ,Mat,Mat &  );
Mat BackwardTransform(Mat , Mat , Mat );
Mat Superimpose_Memory_Images(Mat , Mat, int );
Mat UpdateCompetition(Mat, Mat, Mat ,int , double,Mat,double p = 1);
Mat UpdateCompetition_Memory(Mat, Mat, Mat ,int , double);

Mat UpdateCompetition_Background(Mat, Mat, Mat, Mat, Mat ,int );
Mat UpdateCompetition_Memory_Background(Mat, Mat, Mat, Mat, Mat ,int );

Fwd_Path_Values ForwardTransform_Background(Mat , Mat ,Mat  );
Mat BackwardTransform_Background(Mat , Mat , Mat );
Mat mysolveAffine(vector<Point2f> src, vector<Point2f> dst);

#endif
#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <time.h>       /* time */


using namespace std;
using namespace cv;

#define PI 3.14159265
int acc = 150;		/*number of random number within the range*/
double maxscale = 1;	/*maximum scaling factor*/
double minscale = 0.5;	/*minimum scaling factor*/

/* Out of date function, unused any more*/
Mat Combine_Transform(Mat t1, Mat t2) {
	/*apply t1 and then t2, the combined transformation matrix is returned*/
	Mat newrow = (Mat_<double>(1, 3) << 0, 0, 1);
	Mat temp(3, 3, CV_64F);
	t1.push_back(newrow);
	t2.push_back(newrow);
	cout << t2 << endl << t1 << endl;
	temp = t2*t1;
	cout << temp<<endl;
	Mat result(2,3,CV_64F);
	temp(Rect(0, 0, 3, 2)).copyTo(result(Rect(0, 0, 3, 2)));
	//cout << result << endl;
	return result;
}

/*	Apply random transformation to the source image and return the result
	The transformation is represented as theta/xtranslate/ytranslate/scale variable
	Each random number is generated as uniform distribution across a number (acc) of points within range
	flag = 1 create the random transformation
	flag = 2 will add small random noise to the previous set of transformation
	*/
Mat Rand_Transform(Mat src, double & theta, double & xtranslate, double & ytranslate, double & scale, int flag = 1) {
	srand(time(NULL));
	if (flag==1) {
		theta = double(rand() % acc) / acc * 2 * 90 - 90;		/*the possible theta is -90:180/acc:90*/
		xtranslate = double(rand() % acc) / acc * src.cols*0.8 - round(0.4*src.cols);		/*xtranslate -0.5cols:cols/acc:0.5cols */
		ytranslate = double(rand() % acc) / acc * src.rows*0.8 - round(0.4*src.rows);		/*ytranslate -0.5rows:rows/acc:0.5rows */
		scale = double(rand() % acc) / acc*(maxscale - minscale) + minscale;

		//scale = 0.616;
		//theta = -55;
		//ytranslate = 113;
		//xtranslate = -111;
	}
	else if (flag == 2) {	/* add small noise*/
		theta += double(rand() % acc -acc/2) / acc * 2 * 90 *0.1;		
		xtranslate += double(rand() % acc - acc/2) / acc * src.cols*0.8 *0.15;
		ytranslate += double(rand() % acc - acc / 2) / acc * src.rows*0.8*0.15;
		scale += double(rand() % acc - acc / 2) / acc*(maxscale - minscale)*0.15;
	}
	//scale = 0.616;
	//theta = -55;
	//ytranslate = 113;
	//xtranslate = -111;
	//scale = 0.676;
	//theta = -21;
	//ytranslate = -69;
	//xtranslate = -82;
	//scale = 0.5;
	//theta = -54;
	//ytranslate = -34;
	//xtranslate = 46;
	printf("xtran=%f,ytran=%f,theta=%f,scale=%f\n", xtranslate, ytranslate, theta, scale);

	Mat Rot = getRotationMatrix2D(Point2f(round(src.cols / 2), round(src.rows / 2)), theta, scale);
	Mat Translate = (Mat_<double>(2, 3) << 1, 0, xtranslate, 0, 1, ytranslate);
	//cout << Translate << endl << Rot << endl;
	//Mat T = Combine_Transform(Rot,Translate);
	//cout << T << endl;
	Mat result;
	//cout << Rot << endl;
	warpAffine(src, result, Rot, Size(src.cols, src.rows));
	warpAffine(result, result, Translate, Size(src.cols, src.rows));

	return result;
}



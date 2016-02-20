#include <iostream>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

#define PI 3.14159265
int acc = 1000;		/*accuracy of the image transformation, e.g. step size between random variable*/
int maxscale = 2;	/*maximum scaling factor*/

Mat Combine_Transform(Mat t1, Mat t2) {
	/*apply t1 and then t2, the combined transformation matrix is returned*/
	Mat newrow = (Mat_<double>(1, 3) << 0, 0, 1);
	Mat temp(3, 3, CV_64F);
	t1.push_back(newrow);
	t2.push_back(newrow);
	temp = t2*t1;
	//cout << temp<<endl;
	Mat result(2,3,CV_64F);
	temp(Rect(0, 0, 3, 2)).copyTo(result(Rect(0, 0, 3, 2)));
	//cout << result << endl;
	return result;
}

Mat Rand_Transform(Mat src, double & theta, double & xtranslate, double & ytranslate, double & scale) {
	theta = double(rand() % acc)/acc*2*PI-PI;		/*the possible theta is -pi:2pi/acc:pi*/
	xtranslate = double(rand() % acc)/acc * 2 * src.cols - src.cols;		/*xtranslate -cols:2cols/acc:cols */
	ytranslate = double(rand() % acc)/acc * 2 * src.rows - src.rows;		/*ytranslate -rows:2rows/acc:rows */
	scale = double(rand() % acc)/acc*maxscale;

	Mat Rot = getRotationMatrix2D(Point2f(round(src.cols / 2), round(src.rows / 2)), theta, scale);
	Mat Translate = (Mat_<double>(2, 3) << 1, 0, xtranslate, 0, 1, ytranslate);
	//cout << Translate;
	Mat T = Combine_Transform(Translate,Rot);

	Mat result;
	warpAffine(src, result, T, Size(src.cols, src.rows), 0);

	return result;
}



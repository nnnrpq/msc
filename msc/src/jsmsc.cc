#ifdef WIN32
#pragma warning( push ) 
#pragma warning( disable: 4530 )
namespace std { typedef type_info type_info; }
#endif

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <cstdio>
#include <fstream>
#include <iostream>

#include <node.h>
#include <v8.h>
#include <ArrayBuffer.h>
#include <nan.h>

#include "../src/Image_PreProcessing.h"
#include "../src/Generate_Image.h"
#include "../src/Learn_Image.h"
#include "../src/MSC.h"

using namespace std;
using namespace v8;
using namespace cv;


Size img_size;
Mat cropped_memory_images;
int framectl = 1;
int width;
int height;

void DroneControl(Mat src, Mat mem, Isolate* isolate, Local<Object> &obj) {
	double Thresh_VAL = 100;
	double MAX_VAL = 1; 
	Mat src_gray;
	Mat dst; 
	Mat Fwd_Image;
	Mat Bwd_Image;
	TransformationSet finalTrans;
	finalTrans.nonIdenticalCount = -1;

	//imshow("src", src);
	////
	////imwrite("acquire.jpg", src);
	//waitKey();

	src_gray = src.clone();
	dst.create(src_gray.size(), src_gray.type());

	Mat Edge_Detected_Image_GrayScale = CannyThreshold(src_gray).clone();
	Mat Edge_Detected_Image_unpadded(src_gray.rows, src_gray.cols, CV_32FC1);
	threshold(Edge_Detected_Image_GrayScale, Edge_Detected_Image_unpadded, Thresh_VAL, MAX_VAL, THRESH_BINARY);
	Edge_Detected_Image_unpadded.convertTo(Edge_Detected_Image_unpadded, CV_32FC1);

	Mat mem_img_canny = CannyThreshold_MemoryImages(mem);
	Mat mem_img_edge(mem_img_canny.rows, mem_img_canny.cols, CV_32FC1);
	threshold(mem_img_canny, mem_img_edge, Thresh_VAL, MAX_VAL, THRESH_BINARY);

	 width = src.cols;
	 height = src.rows;
	
	int maxRows = Edge_Detected_Image_unpadded.rows;
	int maxCols = Edge_Detected_Image_unpadded.cols*width/height;

	Mat Memory_Images = padImageMatrix(mem_img_edge, maxRows, maxCols);

	resize(Edge_Detected_Image_unpadded, Edge_Detected_Image_unpadded,Size(maxCols, maxRows));

	Size img_size = Memory_Images.size();

	if (!framectl)
		finalTrans.nonIdenticalCount = -1;
	int ret = SL_MSC(Edge_Detected_Image_unpadded, Memory_Images, img_size, &Fwd_Image, &Bwd_Image, finalTrans);

	


	if (finalTrans.xTranslate<-0.1*maxCols) {
		/* left is positive*/
		obj->Set(v8::String::NewFromUtf8(isolate, "roll"), v8::Number::New(isolate, 0.5));
		cout << "move left" << endl;
		//return;
	}
	else if (finalTrans.xTranslate>0.1*maxCols) {
		obj->Set(v8::String::NewFromUtf8(isolate, "roll"), v8::Number::New(isolate, -0.5));
		cout << "move right" << endl;
	}
	else {
		obj->Set(v8::String::NewFromUtf8(isolate, "roll"), v8::Number::New(isolate, 0));
		cout << "no left/right" << endl;
	}

	if (finalTrans.yTranslate<-0.1*maxRows) {
		/* front is positive*/
		obj->Set(v8::String::NewFromUtf8(isolate, "pitch"), v8::Number::New(isolate, 0.5));
		cout << "move front" << endl;
	}
	else if (finalTrans.yTranslate>0.1*maxRows) {
		obj->Set(v8::String::NewFromUtf8(isolate, "pitch"), v8::Number::New(isolate, -0.5));
		cout << "move back" << endl;
	}
	else {
		obj->Set(v8::String::NewFromUtf8(isolate, "pitch"), v8::Number::New(isolate, 0));
		cout << "no front/back" << endl;
	}
	return;

}

void jsmsc(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = args.GetIsolate();

	//if (!args[0]->IsArrayBuffer()) {
	//	/*isolate->ThrowException(v8::Exception::TypeError(
	//		v8::String::NewFromUtf8(isolate, "Wrong data type~~~~")));*/
	//	cout << "It is not array buffer" << endl;
	//	//args.GetReturnValue().Set(0);
	//	//return;
	//}
	//else {
	//	cout<<"It is array buffer"<<endl;
	//	//args.GetReturnValue().Set(10);
	//}

	//Local<Object> obj = Object::New(isolate);
	//obj->Set(v8::String::NewFromUtf8(isolate, "spin"), v8::Number::New(isolate, -0.3));
	//args.GetReturnValue().Set(obj);
	//return;

	Local<Object> bufferObj = args[0]->ToObject();
	unsigned char* data = (unsigned char*)node::Buffer::Data(bufferObj);
	size_t bufferLength = node::Buffer::Length(bufferObj);
	//cout << bufferLength << endl;
	//cout<<Mat(Size(1280, 720), CV_8UC3, data);
	//cout << data;

	Mat src = Mat(Size(width, height), CV_8UC3);
	src = imdecode(Mat(bufferLength,1,CV_8U,data), CV_LOAD_IMAGE_GRAYSCALE);

	Mat mem = imread("../img_1.png", CV_LOAD_IMAGE_GRAYSCALE);

	//try {
	//	imshow("testimg!", img);
	//}
	//catch (int err) {
	//	cout << "error! " << err << endl;
	//}
	//waitKey();
	//v8::Local<v8::ArrayBuffer> argBuff = node::ArrayBuffer::ToArrayBuffer(isolate);

	Local<Object> obj = Object::New(isolate);
	//CalcReturnVal(src, test, isolate, obj);
	DroneControl(src, mem, isolate, obj);

	args.GetReturnValue().Set(obj);

}

void CreateFunction(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = args.GetIsolate();

	Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, jsmsc);
	Local<Function> fn = tpl->GetFunction();

	// omit this to make it anonymous
	fn->SetName(v8::String::NewFromUtf8(isolate, "Test"));

	args.GetReturnValue().Set(fn);
}
void CalcReturnVal(Mat img, Isolate* isolate, Local<Object> &obj) {
	/*test function, guide the drone to get away from a light*/
	Mat thimg;
	//threshold(img, thimg, 220, 255, CV_THRESH_OTSU);
	threshold(img, thimg, 220, 255, CV_THRESH_BINARY);
	int cols = thimg.cols;
	int left = countNonZero(thimg.colRange(0, round(cols / 2)));
	int right = countNonZero(thimg.colRange(round(cols / 2), round(cols)));
	//imshow("bw", thimg);
	//waitKey();
	//


	if ((left + right)<0.1*cols*thimg.rows) {
		obj->Set(v8::String::NewFromUtf8(isolate, "spin"), v8::Number::New(isolate, 0));
		cout << "stay there" << endl;
		return;
	}
	if (left > right) {
		obj->Set(v8::String::NewFromUtf8(isolate, "spin"), v8::Number::New(isolate, 0.3));
		cout << "Turn clkwise" << endl;
	}
	else if (left<right) {
		obj->Set(v8::String::NewFromUtf8(isolate, "spin"), v8::Number::New(isolate, -0.3));
		cout << "Turn counter clkwise" << endl;
	}
	return;

}

void Init(Local<Object> exports, Local<Object> module) {
	NODE_SET_METHOD(module, "exports", CreateFunction);
}

NODE_MODULE(addon, Init)

#ifdef WIN32
#pragma warning( pop )

#endif
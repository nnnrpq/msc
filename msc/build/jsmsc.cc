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

using namespace std;
using namespace v8;
using namespace cv;

int width = 1280;
int height = 720;

void CalcReturnVal(Mat img, Isolate* isolate, Local<Object> &obj) {
	/*test function, guide the drone to get away from a light*/
	Mat thimg;
	//threshold(img, thimg, 220, 255, CV_THRESH_OTSU);
	threshold(img, thimg, 220, 255, CV_THRESH_BINARY);
	int cols = thimg.cols;
	int left = countNonZero(thimg.colRange(0, round(cols/2)));
	int right = countNonZero(thimg.colRange(round(cols / 2), round(cols)));
	//imshow("bw", thimg);
	//waitKey();
	//


	if ((left+right)<0.1*cols*thimg.rows) {
		obj->Set(v8::String::NewFromUtf8(isolate, "spin"), v8::Number::New(isolate, 0));
		cout << "stay there" << endl;
		return;
	}
	if (left > right) {
		obj->Set(v8::String::NewFromUtf8(isolate, "spin"), v8::Number::New(isolate,0.3));
		cout << "Turn clkwise" << endl;
	}
	else if (left<right){
		obj->Set(v8::String::NewFromUtf8(isolate, "spin"), v8::Number::New(isolate, -0.3));
		cout << "Turn counter clkwise" << endl;
	}
	return;

}
void jsmsc(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = args.GetIsolate();

	if (!args[0]->IsArrayBuffer()) {
		/*isolate->ThrowException(v8::Exception::TypeError(
			v8::String::NewFromUtf8(isolate, "Wrong data type~~~~")));*/
		cout << "It is not array buffer" << endl;
		//args.GetReturnValue().Set(0);
		//return;
	}
	else {
		cout<<"It is array buffer"<<endl;
		//args.GetReturnValue().Set(10);
	}
	//Local<Object> obj = Object::New(isolate);
	//obj->Set(v8::String::NewFromUtf8(isolate, "spin"), v8::Number::New(isolate, -0.3));
	//args.GetReturnValue().Set(obj);
	//return;

	Local<Object> bufferObj = args[0]->ToObject();
	unsigned char* data = (unsigned char*)node::Buffer::Data(bufferObj);
	size_t bufferLength = node::Buffer::Length(bufferObj);
	cout << bufferLength << endl;
	//cout<<Mat(Size(1280, 720), CV_8UC3, data);
	//cout << data;

	Mat img = Mat(Size(width, height), CV_8UC3);
	img = imdecode(Mat(bufferLength,1,CV_8U,data), CV_LOAD_IMAGE_GRAYSCALE);

	//try {
	//	imshow("testimg!", img);
	//}
	//catch (int err) {
	//	cout << "error! " << err << endl;
	//}
	//waitKey();
	//v8::Local<v8::ArrayBuffer> argBuff = node::ArrayBuffer::ToArrayBuffer(isolate);

	Local<Object> obj = Object::New(isolate);
	CalcReturnVal(img, isolate, obj);

	args.GetReturnValue().Set(obj);

}

void CreateFunction(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = args.GetIsolate();

	Local<FunctionTemplate> tpl = FunctionTemplate::New(isolate, jsmsc);
	Local<Function> fn = tpl->GetFunction();

	// omit this to make it anonymous
	fn->SetName(v8::String::NewFromUtf8(isolate, "imTest"));

	args.GetReturnValue().Set(fn);
}

void Init(Local<Object> exports, Local<Object> module) {
	NODE_SET_METHOD(module, "exports", CreateFunction);
}

NODE_MODULE(addon, Init)



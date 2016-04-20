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

using namespace std;
using namespace v8;
using namespace cv;

int width;
int height;

void jsmsc(const FunctionCallbackInfo<Value>& args) {
	Isolate* isolate = args.GetIsolate();

	if (!args.This()->IsArrayBuffer()) {
		isolate->ThrowException(v8::Exception::TypeError(
			v8::String::NewFromUtf8(isolate, "Wrong data type")));
		return;
	}
	else
		args.GetReturnValue().Set(10);

	//v8::Local<v8::ArrayBuffer> argBuff = node::ArrayBuffer::ToArrayBuffer(isolate);

	//Mat img();
}

void Init(Local<Object> exports) {
	NODE_SET_METHOD(exports, "jsmsc", jsmsc);
}

NODE_MODULE(jsmsc, Init)



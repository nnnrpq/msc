#include <node.h>
using namespace v8;

Handle<Value> Add(const Arguments& args)
{
	HandleScope scope;
	// 代表了可以传入 2 个以上的参数，但实际上我们只用前两个
	if (args.Length() < 2)
	{
		// 抛出错误
		ThrowException(Exception::TypeError(String::New("Wrong number of arguments")));
		// 返回空值
		return scope.Close(Undefined());
	}
	// 若前两个参数其中一个不是数字的话
	
	if (!args[0]->IsNumber() || !args[1]->IsNumber())
	{
		// 抛出错误并返回空值
		ThrowException(Exception::TypeError(String::New("Wrong arguments")));
		return scope.Close(Undefined());
	}
	// 具体参考 v8 文档
	//     http://izs.me/v8-docs/classv8_1_1Value.html#a6eac2b07dced58f1761bbfd53bf0e366)
	// 的 `NumberValue` 函数
	Local<Number> num = Number::New(args[0]->NumberValue() + args[1]->NumberValue());
	return scope.Close(num);
}
#include <node.h>
using namespace v8;

Handle<Value> Add(const Arguments& args)
{
	HandleScope scope;
	// �����˿��Դ��� 2 �����ϵĲ�������ʵ��������ֻ��ǰ����
	if (args.Length() < 2)
	{
		// �׳�����
		ThrowException(Exception::TypeError(String::New("Wrong number of arguments")));
		// ���ؿ�ֵ
		return scope.Close(Undefined());
	}
	// ��ǰ������������һ���������ֵĻ�
	
	if (!args[0]->IsNumber() || !args[1]->IsNumber())
	{
		// �׳����󲢷��ؿ�ֵ
		ThrowException(Exception::TypeError(String::New("Wrong arguments")));
		return scope.Close(Undefined());
	}
	// ����ο� v8 �ĵ�
	//     http://izs.me/v8-docs/classv8_1_1Value.html#a6eac2b07dced58f1761bbfd53bf0e366)
	// �� `NumberValue` ����
	Local<Number> num = Number::New(args[0]->NumberValue() + args[1]->NumberValue());
	return scope.Close(num);
}
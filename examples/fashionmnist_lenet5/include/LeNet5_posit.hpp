// General headers
#include <positnn/positnn>

template <typename T>
class LeNet5_posit : public Layer<typename T::Optimizer>{
public:
	LeNet5_posit() :
		conv1(1, 6, 5, 1, 2),
		conv2(6, 16, 5),
		conv3(16, 120, 5),
		fc1(120, 84),
		fc2(84, 10),
		max_pool1(2, 2),
		max_pool2(2, 2)
	{
		this->register_module(conv1);
		this->register_module(conv2);
		this->register_module(conv3);
		this->register_module(fc1);
		this->register_module(fc2);
	}
	
	// Posit precisions
	using O = typename T::Optimizer;
	using F = typename T::Forward;
	using B = typename T::Backward;
	using G = typename T::Gradient;
	
	StdTensor<F> forward(StdTensor<F> x) {
		x = conv1.forward(x);
		x = max_pool1.forward(x);
		x = relu1.forward(x);
		
		x = conv2.forward(x);
		x = max_pool2.forward(x);
		x = relu2.forward(x);
		
		x = conv3.forward(x);
		x = relu3.forward(x);
		
		x.reshape({x.shape()[0], 120});
		
		x = fc1.forward(x);
		x = relu4.forward(x);
		
		x = fc2.forward(x);
		return x;
	}
	
	StdTensor<B> backward(StdTensor<B> x) {
		x = fc2.backward(x);
		
		x = relu4.backward(x);
		x = fc1.backward(x);
		
		x.reshape({x.shape()[0], 120, 1 ,1});
		
		x = relu3.backward(x);
		x = conv3.backward(x);
		
		x = relu2.backward(x);
		x = max_pool2.backward(x);
		x = conv2.backward(x);
		
		x = relu1.backward(x);
		x = max_pool1.backward(x);
		x = conv1.backward(x);
		return x;
	}
	
private:
	Conv2d<O, F, B, G> conv1, conv2, conv3;
	Linear<O, F, B, G> fc1, fc2;
	MaxPool2d<F, B> max_pool1, max_pool2;
	ReLU relu1, relu2, relu3, relu4;
};

#ifndef LENET5_POSIT_HPP
#define LENET5_POSIT_HPP

// Custom headers
#include "positnn/activation/ReLU.hpp"
#include "positnn/layer/BackScale.hpp"
#include "positnn/layer/Conv2d.hpp"
#include "positnn/layer/Layer.hpp"
#include "positnn/layer/Linear.hpp"
#include "positnn/layer/MaxPool2d.hpp"
#include "positnn/tensor/StdTensor.hpp"

template <typename T>
class LeNet5_posit : public Layer<typename T::Optimizer>{
public:
	LeNet5_posit() :
		conv1(1, 6, 5, 1, 2),
		conv2(6, 16, 5),
		conv3(16, 120, 5),
		max_pool1(2, 2),
		max_pool2(2, 2),
		fc1(120, 84),
		fc2(84, 10)
	{
		this->register_module(conv1);
		this->register_module(conv2);
		this->register_module(conv3);
		this->register_module(fc1);
		this->register_module(fc2);
	}

	StdTensor<typename T::Forward> forward(StdTensor<typename T::Forward> x) {
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

	StdTensor<typename T::Backward> backward(StdTensor<typename T::Backward> x) {
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
	Conv2d<typename T::Optimizer, typename T::Forward, typename T::Backward, typename T::Gradient> conv1;
	Conv2d<typename T::Optimizer, typename T::Forward, typename T::Backward, typename T::Gradient> conv2;
	Conv2d<typename T::Optimizer, typename T::Forward, typename T::Backward, typename T::Gradient> conv3;
	MaxPool2d<typename T::Forward, typename T::Backward> max_pool1;
	MaxPool2d<typename T::Forward, typename T::Backward> max_pool2;
	Linear<typename T::Optimizer, typename T::Forward, typename T::Backward, typename T::Gradient> fc1;
	Linear<typename T::Optimizer, typename T::Forward, typename T::Backward, typename T::Gradient> fc2;
	ReLU relu1;
	ReLU relu2;
	ReLU relu3;
	ReLU relu4;
};

#endif /* LENET5_POSIT_HPP */

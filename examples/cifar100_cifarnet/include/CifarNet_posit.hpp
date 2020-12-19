#ifndef CIFARNET_POSIT_HPP
#define CIFARNET_POSIT_HPP

// Custom headers
#include "positnn/activation/ReLU.hpp"
#include "positnn/layer/Conv2d.hpp"
#include "positnn/layer/Dropout.hpp"
#include "positnn/layer/Layer.hpp"
#include "positnn/layer/Linear.hpp"
#include "positnn/layer/MaxPool2d.hpp"
#include "positnn/tensor/StdTensor.hpp"

template <typename T>
class CifarNet_posit : public Layer<typename T::Optimizer>{
public:
	CifarNet_posit(size_t num_classes=100) :
		conv1(3, 8, 5, 1, 2),
		conv2(8, 16, 5, 1, 2),
		max_pool1(2, 2),
		max_pool2(2, 2),
		fc1(1024, 384),
		fc2(384, 192),
		fc3(192, num_classes),
		dropout1(0.5),
		dropout2(0.5)
	{
		this->register_module(conv1);
		this->register_module(conv2);
		this->register_module(fc1);
		this->register_module(fc2);
		this->register_module(fc3);
		this->register_module(dropout1);
		this->register_module(dropout2);
	}
	
	// Posit precisions
	using O = typename T::Optimizer;
	using F = typename T::Forward;
	using B = typename T::Backward;
	using G = typename T::Gradient;

	StdTensor<F> forward(StdTensor<F> x) {
		// Convolutional layers
		x = conv1.forward(x);
		x = max_pool1.forward(x);
		x = relu1.forward(x);

		x = conv2.forward(x);
		x = max_pool2.forward(x);
		x = relu2.forward(x);

		// Flatten
		x.reshape({x.shape()[0], 1024});

		// Fully connected layers
		x = dropout1.forward(x);
		x = fc1.forward(x);
		x = relu3.forward(x);

		x = dropout2.forward(x);
		x = fc2.forward(x);
		x = relu4.forward(x);

		x = fc3.forward(x);
		return x;
	}

	StdTensor<B> backward(StdTensor<B> x) {
		// Fully connected layers
		x = fc3.backward(x);

		x = relu4.backward(x);
		x = fc2.backward(x);
		x = dropout2.backward(x);

		x = relu3.backward(x);
		x = fc1.backward(x);
		x = dropout1.backward(x);
		
		// De-flatten
		x.reshape({x.shape()[0], 16, 8, 8});

		// Convolutional layers

		x = relu2.backward(x);
		x = max_pool2.backward(x);
		x = conv2.backward(x);

		x = relu1.backward(x);
		x = max_pool1.backward(x);
		x = conv1.backward(x);

		return x;
	}

private:
	Conv2d<O, F, B, G> conv1, conv2;
	MaxPool2d<F, B> max_pool1, max_pool2;
	Linear<O, F, B, G> fc1, fc2, fc3;
	Dropout<O> dropout1, dropout2;
	ReLU relu1, relu2, relu3, relu4;
};

#endif /* CIFARNET_POSIT_HPP */

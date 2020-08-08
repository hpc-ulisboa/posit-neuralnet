#ifndef LENET5_POSIT_SCALING_HPP
#define LENET5_POSIT_SCALING_HPP

// Custom headers
#include "positnn/activation/ReLU.hpp"
#include "positnn/layer/Conv2d.hpp"
#include "positnn/layer/Layer.hpp"
#include "positnn/layer/Linear.hpp"
#include "positnn/layer/MaxPool2d.hpp"
#include "positnn/tensor/StdTensor.hpp"

template <typename Posit>
class LeNet5_posit_scaling : public Layer<Posit>{
public:
	using ScalePosit = posit<32, 3>;

	LeNet5_posit_scaling() :
		bs(6),
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

	StdTensor<Posit> forward(StdTensor<Posit> input) {
		input = conv1.forward(input);
		input = max_pool1.forward(input);
		input = relu1.forward(input);

		input = conv2.forward(input);
		input = max_pool2.forward(input);
		input = relu2.forward(input);

		input = conv3.forward(input);
		input = relu3.forward(input);

		input.reshape({input.shape()[0], 120});

		input = fc1.forward(input);
		input = relu4.forward(input);

		input = fc2.forward(input);
		return input;
	}

	StdTensor<Posit> backward(StdTensor<Posit> error) {
		error = bs.backward(5, error);

		error = fc2.backward(error);
		error = bs.backward(4, error, fc2.parameters());

		error = relu4.backward(error);
		error = fc1.backward(error);
		error = bs.backward(3, error, fc1.parameters());

		error.reshape({error.shape()[0], 120, 1 ,1});

		error = relu3.backward(error);
		error = conv3.backward(error);
		error = bs.backward(2, error, conv3.parameters());

		error = relu2.backward(error);
		error = max_pool2.backward(error);
		error = conv2.backward(error);
		error = bs.backward(1, error, conv2.parameters());

		error = relu1.backward(error);
		error = max_pool1.backward(error);
		error = conv1.backward(error);
		error = bs.backward(0, error, conv1.parameters());

		return error;
	}

	BackScale<ScalePosit> bs;

private:
	Conv2d<Posit> conv1;
	Conv2d<Posit> conv2;
	Conv2d<Posit> conv3;
	ReLU<Posit> relu1;
	ReLU<Posit> relu2;
	ReLU<Posit> relu3;
	ReLU<Posit> relu4;
	MaxPool2d<Posit> max_pool1;
	MaxPool2d<Posit> max_pool2;
	Linear<Posit> fc1;
	Linear<Posit> fc2;
};

#endif /* LENET5_POSIT_SCALING_HPP */

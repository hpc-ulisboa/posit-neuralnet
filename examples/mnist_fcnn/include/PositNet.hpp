#ifndef NET_HPP
#define NET_HPP

// Custom headers
#include "positnn/activation/ReLU.hpp"
//#include "positnn/layer/BatchNorm1d.hpp"
#include "positnn/layer/Layer.hpp"
#include "positnn/layer/Linear.hpp"
//#include "positnn/layer/RangeBatchNorm1d.hpp"
#include "positnn/tensor/StdTensor.hpp"

template <typename T>
class PositNet : public Layer<typename T::Optimizer>{
public:
	PositNet() :
		linear1(784, 32),
		linear2(32, 10)
		//batch_norm1(32),
	{
		this->register_module(linear1);
		this->register_module(linear2);
		//this->register_module(batch_norm1);
	}

	StdTensor<typename T::Forward> forward(StdTensor<typename T::Forward> x) {
		// Flatten data
		x.reshape({x.shape()[0], 784});

		x = linear1.forward(x);
		//x = batch_norm1.forward(x);
		x = relu.forward(x);
		x = linear2.forward(x);
		return x;
	}

	StdTensor<typename T::Backward> backward(StdTensor<typename T::Backward> x) {
		x = linear2.backward(x);
		x = relu.backward(x);
		//x = batch_norm1.backward(x);
		x = linear1.backward(x);
		return x;
	}

private:
	Linear<typename T::Optimizer, typename T::Forward, typename T::Backward, typename T::Gradient> linear1;
	Linear<typename T::Optimizer, typename T::Forward, typename T::Backward, typename T::Gradient> linear2;
	//BatchNorm1d<Posit> batch_norm1;
	//RangeBatchNorm1d<Posit> batch_norm1;
	ReLU relu;
};

#endif /* NET_HPP */

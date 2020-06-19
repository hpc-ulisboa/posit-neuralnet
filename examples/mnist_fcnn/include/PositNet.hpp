#ifndef NET_HPP
#define NET_HPP

// Custom headers
#include "positnn/activation/Sigmoid.hpp"
//#include "positnn/layer/BatchNorm1d.hpp"
#include "positnn/layer/Layer.hpp"
#include "positnn/layer/Linear.hpp"
//#include "positnn/layer/RangeBatchNorm1d.hpp"
#include "positnn/tensor/StdTensor.hpp"

template <typename Posit>
class PositNet : public Layer<Posit>{
public:
	PositNet() :
		linear1(784, 32),
		//batch_norm1(32),
		linear2(32, 10)
	{
		this->register_module(linear1);
		//this->register_module(batch_norm1);
		this->register_module(linear2);
	}

	StdTensor<Posit> forward(StdTensor<Posit>& input) {
		StdTensor<Posit> output = linear1.forward(input);
		//output = batch_norm1.forward(output);
		output = sigmoid1.forward(output);
		return linear2.forward(output);
	}

	StdTensor<Posit> backward(StdTensor<Posit>& output_error) {
		StdTensor<Posit> error = linear2.backward(output_error);
		error = sigmoid1.backward(error);
		//error = batch_norm1.backward(error);
		return linear1.backward(error);
	}

private:
	Linear<Posit> linear1;
	//BatchNorm1d<Posit> batch_norm1;
	//RangeBatchNorm1d<Posit> batch_norm1;
	Sigmoid<Posit> sigmoid1;
	Linear<Posit> linear2;
};

#endif /* NET_HPP */

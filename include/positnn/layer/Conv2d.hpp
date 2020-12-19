#ifndef CONV2D_HPP
#define CONV2D_HPP

// General headers
#include <cmath>

// Custom headers
#include "init.hpp"
#include "Layer.hpp"
#include "../tensor/convolution.hpp"
#include "../tensor/MixedTensor.hpp"
#include "../tensor/sum.hpp"
#include "../tensor/StdTensor.hpp"

template <typename OptimizerT, typename ForwardT=OptimizerT, typename BackwardT=ForwardT, typename GradientT=BackwardT>
class Conv2d : public Layer<OptimizerT> {
// TODO: implement dilation
public:
	Conv2d(size_t _in_channels, size_t _out_channels, size_t _kernel_size, size_t _stride=1, size_t _padding=0, size_t _dilation=1) :
		in_channels(_in_channels),
		out_channels(_out_channels),
		kernel_size(_kernel_size),
		stride(_stride),
		padding(_padding),
		dilation(_dilation),
		weight({out_channels, in_channels, kernel_size, kernel_size}),
		bias(out_channels),
		weight_gradient({out_channels, in_channels, kernel_size, kernel_size}),
		bias_gradient(out_channels)
	{
		this->register_parameter(weight, weight_gradient);
		this->register_parameter(bias, bias_gradient);

		reset_parameters();
	}

	void reset_parameters() {
		std::cerr << "Conv2d layer is not initialized" << std::endl;
	}

	template <typename T>
	StdTensor<ForwardT> forward(StdTensor<T> const& x) {
		input = x;
		return convolution2d<ForwardT::nbits, ForwardT::es>(x, weight.get_forward(), bias.get_forward(), stride, padding, 1, dilation, &w1);
	}

	template <typename T>
	StdTensor<BackwardT> backward(StdTensor<T> const& delta) {
		gradient(delta);
		StdTensor<BackwardT> rotated = rotate_weight(weight.get_backward());
		return convolution2d<BackwardT::nbits, BackwardT::es>(delta, rotated, StdTensor<BackwardT>(), 1, (kernel_size-1)*dilation-padding, stride, dilation, &w3);
	}

	void gradient(StdTensor<GradientT> const& delta) {
		StdTensor<GradientT> temp_weight_gradient = convolution2d_gradient(input, delta, stride, padding, dilation, &w2);
		StdTensor<GradientT> temp_bias_gradient = sum_last2(delta);

		// If there are many samples
		if(input.dim()>1 && input.shape()[0]>1){
			temp_weight_gradient /= input.shape()[0];

			temp_bias_gradient = sum_first(temp_bias_gradient);
			temp_bias_gradient /= input.shape()[0];
		}

		weight_gradient += temp_weight_gradient;
		bias_gradient += temp_bias_gradient;

		return;
	}

private:
	size_t in_channels;
	size_t out_channels;
	size_t kernel_size;
	size_t stride;
	size_t padding;
	size_t dilation;
	MixedTensor<OptimizerT, ForwardT, BackwardT> weight;
	MixedTensor<OptimizerT, ForwardT> bias;
	StdTensor<GradientT> input;
	StdTensor<OptimizerT> weight_gradient;
	StdTensor<OptimizerT> bias_gradient;
	Window w1, w2, w3;
};

#endif /* CONV2D_HPP */

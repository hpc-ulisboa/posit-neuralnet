#ifndef CONV2D_HPP
#define CONV2D_HPP

// General headers
#include <cmath>

// Custom headers
#include "init.hpp"
#include "Layer.hpp"
#include "../tensor/convolution.hpp"
#include "../tensor/StdTensor.hpp"

template <typename Posit>
class Conv2d : public Layer<Posit> {
// TODO: include other options such as padding, dilation, groups, bias, etc
public:
	Conv2d(size_t _in_channels, size_t _out_channels, size_t _kernel_size, size_t _stride=1, size_t _padding=0) :
		in_channels(_in_channels),
		out_channels(_out_channels),
		kernel_size(_kernel_size),
		stride(_stride),
		padding(_padding),
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

	}

	StdTensor<Posit> forward(StdTensor<Posit>& x) {
		input = x;
		return convolution2d(x, weight, bias, stride, padding, &w1);
	}

	StdTensor<Posit> backward(StdTensor<Posit>& delta) {
		gradient(delta);
		StdTensor<Posit> rotated = rotate_weight(weight);
		// TODO: Pass empty bias, correct stride and padding
		return convolution2d(delta, rotated, StdTensor<Posit>(), stride, kernel_size-1, &w3);
	}

	void gradient(StdTensor<Posit>& delta) {
		StdTensor<Posit> temp_weight_gradient = convolution2d_gradient(input, delta, stride,padding, &w2);
		StdTensor<Posit> temp_bias_gradient = sum_last2(delta);

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
	StdTensor<Posit> weight;
	StdTensor<Posit> bias;
	StdTensor<Posit> input;
	StdTensor<Posit> weight_gradient;
	StdTensor<Posit> bias_gradient;
	Window w1, w2, w3;
};

#endif /* CONV2D_HPP */

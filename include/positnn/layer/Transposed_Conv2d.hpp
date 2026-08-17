#ifndef TRANSPOSED_CONV2D_HPP
#define TRANSPOSED_CONV2D_HPP

// General headers
#include <cmath>
#include <iostream>

// Custom headers
#include "init.hpp"
#include "Layer.hpp"
#include "../tensor/transposed_convolution.hpp"
#include "../tensor/convolution.hpp"
#include "../tensor/MixedTensor.hpp"
#include "../tensor/sum.hpp"
#include "../tensor/StdTensor.hpp"
#include "../tensor/Window.hpp"

template <typename OptimizerT, typename ForwardT=OptimizerT, typename BackwardT=ForwardT, typename GradientT=BackwardT>
class TransposedConv2d : public Layer<OptimizerT> {
public:
	TransposedConv2d(size_t _in_channels, size_t _out_channels, size_t _kernel_size, size_t _stride=1, size_t _padding=0, size_t _output_padding=0, size_t _dilation=1) :
		in_channels(_in_channels),
		out_channels(_out_channels),
		kernel_size(_kernel_size),
		stride(_stride),
		padding(_padding),
		output_padding(_output_padding),
		dilation(_dilation),
		// Weight layout is (in_channels, out_channels, kernel_h, kernel_w), i.e. the
		// first two dimensions are swapped compared to Conv2d.
		weight({_in_channels, _out_channels, _kernel_size, _kernel_size}),
		bias(_out_channels),
		weight_gradient({_in_channels, _out_channels, _kernel_size, _kernel_size}),
		bias_gradient(_out_channels)
	{
		this->register_parameter(weight, weight_gradient);
		this->register_parameter(bias, bias_gradient);

		reset_parameters();
	}

	// Same initialization as torch.nn.ConvTranspose2d, drawn into the optimizer
	// copy and propagated by MixedTensor::update(), as in Conv2d and Linear.
	// Note that fan_in here uses dimension 1 of the weight, which for a transposed
	// convolution is out_channels, matching PyTorch.
	void reset_parameters() {
		kaiming_uniform<OptimizerT, float>(weight.get_optimizer(), std::sqrt(5));

		if(!bias.get_optimizer().empty()) {
			float fan_in = calculate_correct_fan<OptimizerT, float>(weight.get_optimizer(), Mode::fan_in);
			float bound = 1.0 / std::sqrt(fan_in);
			set_uniform<OptimizerT, float>(bias.get_optimizer(), -bound, bound);
		}

		weight.update();
		bias.update();
	}

	template <typename T>
	StdTensor<ForwardT> forward(StdTensor<T> const& x) {
		input = x;
		// The layer's output_padding is square, like its other spatial parameters,
		// so the same value goes to both axes.
		return transposed_convolution2d<ForwardT::nbits, ForwardT::es>(
			x, weight.get_forward(), bias.get_forward(),
			stride, padding, output_padding, output_padding, dilation, &w_forward
		);
	}

	template <typename T>
	StdTensor<BackwardT> backward(StdTensor<T> const& delta) {
		gradient(delta);
		return transposed_convolution2d_backward_input<BackwardT::nbits, BackwardT::es>(
			delta, weight.get_backward(),
			input.shape()[2], input.shape()[3],
			stride, padding, dilation, &w_grad_input
		);
	}

	void gradient(StdTensor<GradientT> const& delta) {
		StdTensor<GradientT> temp_weight_gradient = transposed_convolution2d_gradient<GradientT::nbits, GradientT::es>(
			input, delta, kernel_size, stride, padding, dilation, &w_grad_weight
		);

		StdTensor<GradientT> temp_bias_gradient;

		if(input.dim() > 1 && input.shape()[0] > 1){
			temp_bias_gradient = sum_first(delta);
			temp_bias_gradient = sum_last2(temp_bias_gradient);
		}else{
			temp_bias_gradient = sum_last2(delta);
		}

		weight_gradient += temp_weight_gradient;
		bias_gradient   += temp_bias_gradient;

		return;
	}

private:
	size_t in_channels;
	size_t out_channels;
	size_t kernel_size;
	size_t stride;
	size_t padding;
	size_t output_padding;
	size_t dilation;
	MixedTensor<OptimizerT, ForwardT, BackwardT> weight;
	MixedTensor<OptimizerT, ForwardT> bias;
	StdTensor<GradientT> input;
	StdTensor<OptimizerT> weight_gradient;
	StdTensor<OptimizerT> bias_gradient;
	// Gather maps, built once per geometry and then reused
	Window w_forward, w_grad_input, w_grad_weight;
};

#endif /* TRANSPOSED_CONV2D_HPP */

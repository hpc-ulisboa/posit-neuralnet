#ifndef CONV2D_HPP
#define CONV2D_HPP

// General headers
#include <cmath>
#include <vector>

// Custom headers
#include "init.hpp"
#include "Layer.hpp"
#include "../tensor/convolution.hpp"
#include "../tensor/transposed_convolution.hpp"
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

	// Same initialization as torch.nn.Conv2d, and the same shape as Linear's:
	// draw into the optimizer copy, which is the one the optimizer actually owns,
	// then let MixedTensor::update() derive the forward and backward copies. The
	// bounds are computed in float so that std::sqrt can be used.
	void reset_parameters() {
		kaiming_uniform<OptimizerT, float>(weight.get_optimizer(), std::sqrt(5));

		if (!bias.get_optimizer().empty()) {
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
		saved_input_shape = input.shape();
		return convolution2d<ForwardT::nbits, ForwardT::es>(x, weight.get_forward(), bias.get_forward(), stride, padding, 1, dilation, &w1);
	}

	template <typename T>
	StdTensor<BackwardT> backward(StdTensor<T> const& delta) {
		gradient(delta);

		// The gradient w.r.t. the input of a convolution is a transposed convolution
		// of delta with the same weight, so no manual weight rotation is needed.
		//
		// Fix: the previous convolution2d + rotate_weight path zero-padded the border
		// rows that the forward floor division drops when stride > 1, even though those
		// input positions do contribute to an output. That zeroed real gradient and
		// corrupted the gradients of all preceding layers. output_padding recovers the
		// dropped rows and reproduces the original input size exactly.
		if (saved_input_shape.empty())
			throw std::runtime_error("[Conv2d] backward() called before forward()");

		// Per axis: the kernel, stride and padding are square, but the input need
		// not be, and the floor division can drop a row without dropping a column.
		// A 6x7 input at kernel 3, stride 2 needs output_padding 1 on the height
		// and 0 on the width.
		auto op_for = [&](size_t out_size, size_t in_size) {
			size_t const base = (out_size-1)*stride - 2*padding + dilation*(kernel_size-1) + 1;
			return in_size - base;
		};
		size_t const output_padding_h = op_for(delta.shape()[2], saved_input_shape[2]);
		size_t const output_padding_w = op_for(delta.shape()[3], saved_input_shape[3]);

		return transposed_convolution2d<BackwardT::nbits, BackwardT::es>(
			delta, weight.get_backward(), StdTensor<BackwardT>(),
			stride, padding, output_padding_h, output_padding_w, dilation, &w3
		);
	}

	void gradient(StdTensor<GradientT> const& delta) {
		// Fix: when the forward floor division is not exact (e.g. kernel 3, stride 2),
		// convolution2d_gradient returns a kernel gradient that is larger than the
		// kernel (4x4 instead of 3x3). Since operator+= works on the flat index, that
		// silently corrupts the accumulated weight gradient. The correct gradient is
		// the top-left kernel_size x kernel_size block.
		StdTensor<GradientT> temp_weight_gradient =
			crop_weight_gradient(convolution2d_gradient(input, delta, stride, padding, dilation, &w2),
								 kernel_size, kernel_size);
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
	std::vector<size_t> saved_input_shape;
};

#endif /* CONV2D_HPP */

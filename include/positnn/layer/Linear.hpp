#ifndef LINEAR_HPP
#define LINEAR_HPP

// General headers
#include <cmath>

// Custom headers
#include "init.hpp"
#include "Layer.hpp"
#include "../tensor/matrix.hpp"
#include "../tensor/MixedTensor.hpp"
#include "../tensor/sum.hpp"
#include "../tensor/StdTensor.hpp"

template <typename OptimizerT, typename ForwardT=OptimizerT, typename BackwardT=ForwardT, typename GradientT=BackwardT>
class Linear : public Layer<OptimizerT> {
// TODO: permit linear layer with no bias
public:
	Linear(size_t in, size_t out) :
		weight({out, in}),
		bias(out),
		weight_gradient({out, in}),
		bias_gradient(out)
	{
		this->register_parameter(weight, weight_gradient);
		this->register_parameter(bias, bias_gradient);

		reset_parameters();
	}

	void reset_parameters() {
		kaiming_uniform<OptimizerT, float>(weight.get_optimizer(), std::sqrt(5));
		
		float bound = calculate_correct_fan<OptimizerT, float>(weight.get_optimizer(), Mode::fan_in);
		bound = 1.0 / std::sqrt(bound);
		set_uniform<OptimizerT, float>(bias.get_optimizer(), -bound, bound);
	}

	template <typename OtherT>
	StdTensor<ForwardT> forward(StdTensor<OtherT> const& x) {
		input = x;
		return matmul_row_add<ForwardT::nbits, ForwardT::es>(x, weight.get_forward(), bias.get_forward());
	}

	template <typename OtherT>
	StdTensor<BackwardT> backward(StdTensor<OtherT> const& delta) {
		gradient(delta);
		return matmul<BackwardT::nbits, BackwardT::es>(delta, weight.get_backward());
	}

	void gradient(StdTensor<GradientT> const& delta) {
		StdTensor<GradientT> temp_weight_gradient = matmul_col(delta, input);
		StdTensor<GradientT> temp_bias_gradient = delta;

		if(input.dim()>1 && input.shape()[0]>1){
		// If there are many samples
			temp_weight_gradient /= input.shape()[0];

			temp_bias_gradient = sum_first(temp_bias_gradient);
			temp_bias_gradient /= input.shape()[0];
		}

		weight_gradient += temp_weight_gradient;
		bias_gradient += temp_bias_gradient;

		return;
	}

private:
	MixedTensor<OptimizerT, ForwardT, BackwardT> weight;
	MixedTensor<OptimizerT, ForwardT> bias;
	StdTensor<GradientT> input;
	StdTensor<OptimizerT> weight_gradient;
	StdTensor<OptimizerT> bias_gradient;
};

#endif /* LINEAR_HPP */

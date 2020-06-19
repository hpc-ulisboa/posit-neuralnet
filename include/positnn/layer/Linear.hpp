#ifndef LINEAR_HPP
#define LINEAR_HPP

// General headers
#include <cmath>

// Custom headers
#include "init.hpp"
#include "Layer.hpp"
#include "../tensor/matrix.hpp"
#include "../tensor/sum.hpp"
#include "../tensor/StdTensor.hpp"

template <typename Posit>
class Linear : public Layer<Posit> {
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
		kaiming_uniform<Posit, float>(weight, std::sqrt(5));
		
		float bound = calculate_correct_fan<Posit, float>(weight, Mode::fan_in);
		bound = 1.0 / std::sqrt(bound);
		set_uniform<Posit, float>(bias, -bound, bound);
	}

	StdTensor<Posit> forward(StdTensor<Posit>& x) {
		input = x;
		return matmul_row_add(input, weight, bias);
	}

	StdTensor<Posit> backward(StdTensor<Posit>& delta) {
		gradient(delta);
		return matmul(delta, weight);
	}

	void gradient(StdTensor<Posit>& delta) {
		StdTensor<Posit> temp_weight_gradient = matmul_col(delta, input);
		// TODO: it's slow here ^
		StdTensor<Posit> temp_bias_gradient = delta;


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
	StdTensor<Posit> weight;
	StdTensor<Posit> bias;
	StdTensor<Posit> input;
	StdTensor<Posit> weight_gradient;
	StdTensor<Posit> bias_gradient;
};

#endif /* LINEAR_HPP */

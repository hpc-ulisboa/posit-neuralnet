#ifndef INIT_HPP
#define INIT_HPP

// General headers
#include <chrono> 
#include <cmath>
#include <random>

// Custom headers
#include "../tensor/StdTensor.hpp"

template <typename T, typename randT=float>
void set_randn(StdTensor<T>& a, const float mean=0, const float stddev=1){
	size_t i;
	size_t size = a.size();

	std::default_random_engine generator;
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	std::normal_distribution<randT> distribution(mean, stddev);

    for (i=0; i<size; i++) {
		a[i] = T(distribution(generator));
	}
}

template <typename T, typename randT=float>
void set_uniform(StdTensor<T>& a, const float lb=0, const float ub=1){
	size_t i;
	size_t size = a.size();

	std::default_random_engine generator;
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<randT> distribution(lb, ub);

    for (i=0; i<size; i++) {
		a[i] = T(distribution(generator));
	}
}

enum class Mode {fan_in, fan_out};
enum class NonLinearity {linear, conv1d, conv2d, conv3d, conv_transpose1d, conv_transpose2d, conv_transpose3d, sigmoid, tanh, relu, leaky_relu};

template <typename T, typename randT>
randT calculate_correct_fan(StdTensor<T>& tensor, const Mode mode) {
    size_t num_input_fmaps = tensor.shape()[0];
    size_t num_output_fmaps = tensor.shape()[1];
    size_t receptive_field_size = 1;

	// TODO: SETUP FOR MORE THAN 2 DIMENSIONS
	/*
    if (tensor.dim() > 2) {
        receptive_field_size = tensor[0][0].numel()
	}
	*/

	size_t fan;

	if(mode==Mode::fan_in)
		fan = num_input_fmaps * receptive_field_size;
	else
    	fan = num_output_fmaps * receptive_field_size;

    return randT(fan);
}

template <typename randT>
randT calculate_gain(const NonLinearity nonlinearity, const randT param=0.01) {
	if(nonlinearity==NonLinearity::leaky_relu) {
		return randT(std::sqrt(2.0 / (1 + param*param)));
	}
	return randT(1);
}

template <typename T, typename randT>
void kaiming_uniform(StdTensor<T>& tensor, const randT a=0,
		const Mode mode=Mode::fan_in, const NonLinearity nonlinearity=NonLinearity::leaky_relu){
// Copied from PyTorch
// https://github.com/pytorch/pytorch/blob/506996c77e94a17b12bf2f1173d452d98756653e/torch/nn/init.py#L352

	randT fan = calculate_correct_fan<T, randT>(tensor, mode);
	randT gain = calculate_gain<randT>(nonlinearity, a);
	randT std = gain / std::sqrt(fan);
	randT bound = std::sqrt(3.0) * std;

	set_uniform<T, randT>(tensor, -bound, bound);

	return;
}

#endif /* INIT_HPP */

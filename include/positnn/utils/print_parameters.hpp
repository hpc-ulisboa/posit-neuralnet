#ifndef PRINT_PARAMETERS
#define PRINT_PARAMETERS

// General headers
#include <iostream>

#ifdef USING_PYTORCH	// Only compiles the code bellow if PyTorch is available
#include <torch/torch.h>
#endif /* USING_PYTORCH */

// Custom headers
#include "../layer/Parameter.hpp"
#include "../tensor/StdTensor.hpp"

template <typename T>
void print_parameters(std::vector<Parameter<T>>& parameters, const bool transpose_option=false) {
	for(const Parameter<T>& p : parameters){
		std::cout << p.weight << std::endl;
	}
}

#ifdef USING_PYTORCH	// Only compiles the code bellow if PyTorch is available

void print_parameters(std::vector<torch::Tensor> parameters) {
	for (const torch::Tensor& p : parameters) {
		std::cout << p << std::endl;
	}
}

template <typename T=float>
void print_parameters_line(std::vector<torch::Tensor> parameters) {
	for (const torch::Tensor& p : parameters) {
		T* data = p.data_ptr<T>();
		for(size_t i=0, size=p.numel(); i<size; i++) {
			std::cout << data[i] << ' ';
		}
		std::cout << std::endl << std::endl;
	}
}

#endif /* USING_PYTORCH */

#endif /* PRINT_PARAMETERS */

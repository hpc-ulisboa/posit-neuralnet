#ifndef CONVERT_HPP
#define CONVERT_HPP

#ifdef USING_PYTORCH	// Only compiles the code bellow if PyTorch is available

// General headers
#include <torch/torch.h>

// Custom headers
#include "StdTensor.hpp"

template <typename CType, typename CustomType>
StdTensor<CustomType> Tensor_to_StdTensor(torch::Tensor& x) {
	std::vector<size_t> shape(x.sizes().begin(), x.sizes().end());
	StdTensor<CustomType> y(shape);

	auto x_data = x.data_ptr<CType>();

	for(size_t i=0, size=y.size(); i<size; i++)
		y[i] = CustomType(x_data[i]);

	return y;
}

template <typename CType, at::ScalarType TensorType, typename CustomType>
torch::Tensor StdTensor_to_Tensor(StdTensor<CustomType>& x) {
	std::vector<long> shape(x.shape().begin(), x.shape().end());	// TODO: IMPROVE THIS CONVERSION
	torch::Tensor y = torch::empty(shape, TensorType);
	auto y_data = y.data_ptr<CType>();

	for(size_t i=0, size=x.size(); i<size; i++)
		y_data[i] = CType(x[i]);

	return y;
}

template<typename FromType=float, typename Posit>
void copy_parameters(std::vector<torch::Tensor> from, std::vector<Parameter<Posit>>& to) {
	for(size_t i=0, size=to.size(); i<size; i++) {
		to[i].weight = Tensor_to_StdTensor<FromType, Posit>(from[i]);
	}

	return;
}

template<typename T>
void copy_parameters(std::vector<Parameter<T>>& from, std::vector<Parameter<T>>& to) {
	for(size_t i=0, size=to.size(); i<size; i++) {
		to[i].weight = from[i].weight;
	}

	return;
}

#endif /* USING_PYTORCH */

#endif /* CONVERT_HPP */

/*
template <typename From, typename To>
struct ConvertImpl : torch::nn::Module {
	ConvertImpl() {}

	std::vector<To> forward(std::vector<From> x) {
		std::vector<To> y(x.size());
		for(std::vector<int>::size_type i = 0; i != x.size(); i++)
			y[i] = To(x[i]);
		return y;
	}
};

template <typename From, typename To>
class Convert : public torch::nn::ModuleHolder<ConvertImpl<From, To>> {
public:                                                        
	using torch::nn::ModuleHolder<ConvertImpl<From, To>>::ModuleHolder;
};
*/
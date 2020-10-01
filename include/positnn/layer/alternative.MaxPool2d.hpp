#ifndef MAXPOOL2D_HPP
#define MAXPOOL2D_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
//#include "Layer.hpp"
#include "../tensor/maximumpool.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename ForwardT, typename BackwardT=ForwardT>
class MaxPool2d {
//class MaxPool2d : public Layer<Posit> {
public:
	MaxPool2d(size_t _kernel_size, size_t _stride=0, size_t _padding=0) :
		kernel_size(_kernel_size),
		padding(_padding)
	{ 
		stride = (_stride==0) ? _kernel_size : _stride;
	}

	StdTensor<ForwardT> forward(StdTensor<ForwardT> const& x) {
		input_shape = x.shape();
		return maximumpool2d(x, kernel_size, stride, padding, &max_idx, &w);
	}

	StdTensor<BackwardT> backward(StdTensor<BackwardT> const& deltaN) {
		return maximumpool2d_backward(deltaN, input_shape, kernel_size, stride, max_idx);
	}

private:
	size_t kernel_size;
	size_t stride;
	size_t padding;
	std::vector<size_t> input_shape;
	Window w;
	std::vector<size_t> max_idx;
};

#endif /* MAXPOOL2D_HPP */

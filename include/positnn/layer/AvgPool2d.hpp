#ifndef AVGPOOL2D_HPP
#define AVGPOOL2D_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
//#include "Layer.hpp"
#include "../tensor/averagepool.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename ForwardT, typename BackwardT=ForwardT>
class AvgPool2d {
//class AvgPool2d : public Layer<Posit> {
public:
	AvgPool2d(size_t _kernel_size, size_t _stride, size_t _padding=0) :
		kernel_size(_kernel_size),
		padding(_padding)
	{ 
		if(_stride==0)
			stride = _kernel_size;
	}

	StdTensor<ForwardT> forward(StdTensor<ForwardT> const& x) {
		input_shape = x.shape();
		return averagepool2d(x, kernel_size, stride, padding, &w1);
	}

	StdTensor<BackwardT> backward(StdTensor<BackwardT> const& delta) {
		// set deltaN_1 by blocks to the value of delta
		return averagepool2d_backward(delta, input_shape, kernel_size, stride, padding, &w2);
	}

private:
	size_t kernel_size;
	size_t stride;
	size_t padding;
	std::vector<size_t> input_shape;
	Window w1, w2;
};

#endif /* AVGPOOL2D_HPP */

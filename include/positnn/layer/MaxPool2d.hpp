#ifndef MAXPOOL2D_HPP
#define MAXPOOL2D_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "Layer.hpp"
#include "../tensor/maximumpool.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename Posit>
class MaxPool2d : public Layer<Posit> {
public:
	MaxPool2d(size_t _kernel_size, size_t _stride=0, size_t _padding=0) :
		kernel_size(_kernel_size),
		padding(_padding)
	{ 
		stride = (_stride==0) ? _kernel_size : _stride;
	}

	StdTensor<Posit> forward(StdTensor<Posit>& x) {
		input_shape = x.shape();
		return maximumpool2d(x, kernel_size, stride, padding, &w, &max_idx);
	}

	StdTensor<Posit> backward(StdTensor<Posit>& deltaN) {
		StdTensor<Posit> deltaN_1(input_shape);

		for(size_t i=0, size=max_idx.size(); i<size; i++){
			deltaN_1[max_idx[i]] = deltaN[i];
		}

		return deltaN_1;
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

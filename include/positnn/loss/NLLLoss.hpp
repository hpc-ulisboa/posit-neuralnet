#ifndef NLLLOSS_HPP
#define NLLLOSS_HPP

// Custom headers
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename T, typename targetT, typename lossT=float>
class nll_loss : public Loss<T, lossT>{
public:
	nll_loss() { }

	nll_loss(StdTensor<T>& output, StdTensor<targetT>& _target, Reduction reduction=Reduction::Mean) :
		output_shape(output.shape()),
		target(target)
	{
		// TODO: protect if target is not integer
		size_t const rows = output_shape[0];
		size_t const cols = output_shape[1];

		// TODO: throw error if rows != target size
		// TODO: protect for dim!=2

		for(size_t i=0, j=0; i<rows; i++, j+=cols){
			// Calculate loss
			this->loss -= lossT( output[j+target[i]] );
		}

		if(reduction == Reduction::Mean)
			this->loss /= rows;
	}

	StdTensor<T> derivative() {
		StdTensor<T> dloss(output_shape);
		T minusOne(-1);

		size_t const rows = output_shape[0];
		size_t const cols = output_shape[1];

		for(size_t i=0, j=0; i<rows; i++, j+=cols){
			// Calculate derivative of loss
			dloss[j+target[i]] = minusOne;
		}

		return dloss;
	}

private:
	std::vector<size_t> output_shape;
	StdTensor<targetT> target;
};

#endif /* NLLLOSS_HPP */

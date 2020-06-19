#ifndef MSELoss_HPP
#define MSELoss_HPP

// Custom headers
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename T, typename targetT, typename lossT=float>
class mse_loss : public Loss<T, lossT>{
public:
	mse_loss() { }

	mse_loss(StdTensor<T>& _output, StdTensor<targetT>& _target, Reduction reduction=Reduction::Mean) :
		output(_output),
		target(_target)	
	{
		size_t const size = output.size();

		if(size != target.size())
			throw std::invalid_argument( "vectors size differ" );

		for(size_t i=0; i<size; i++){
			// Calculate loss
			targetT error = targetT(output[i]) - target[i];
			this->loss += lossT(error*error);
		}

		if(reduction == Reduction::Mean)
			this->loss /= size;
	}

	StdTensor<T> derivative() {
		StdTensor<T> dloss(output);

		size_t const size = output.size();

		for(size_t i=0; i<size; i++){
			// Calculate derivative of loss
			dloss[i] -= T(target[i]);
			dloss[i] *= 2;
		}

		return dloss;
	}

private:
	StdTensor<T> output;
	StdTensor<targetT> target;
};

#endif /* MSELoss_HPP */

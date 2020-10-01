#ifndef MSELoss_HPP
#define MSELoss_HPP

// Custom headers
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <class ForwardT, class BackwardT=ForwardT, typename lossT=float>
class mse_loss : public Loss<BackwardT, lossT>{
public:
	mse_loss() { }

	mse_loss(StdTensor<ForwardT> const& output, StdTensor<ForwardT> const& target, Reduction reduction=Reduction::Mean) :
		error(output.shape())
	{
		size_t const size = output.size();

		if(size != target.size())
			throw std::invalid_argument( "vectors size differ" );

		for(size_t i=0; i<size; i++){
			// Calculate loss
			ForwardT error_forward = output[i] - target[i];
			this->loss += lossT(error_forward * error_forward);
			
			// Copy to be used in backward
			error[i] = error_forward;
		}

		if(reduction == Reduction::Mean)
			this->loss /= size;
	}

	StdTensor<BackwardT> derivative() {
		StdTensor<BackwardT> dloss = error * 2;

		/*
		for(size_t i=0, size=dloss.size(); i<size; i++){
			// Calculate derivative of loss
			dloss[i] = output[i] - target[i];
			dloss[i] *= 2;
		}
		*/

		return dloss;
	}

private:
	StdTensor<BackwardT> error;
};

#endif /* MSELoss_HPP */

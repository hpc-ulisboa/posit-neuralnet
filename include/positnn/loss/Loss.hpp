#ifndef LOSS_HPP
#define LOSS_HPP

// Custom headers
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

enum class Reduction {Mean, Sum};

template <typename T, typename lossT=float>
class Loss{
public:
	Loss() {}
	virtual ~Loss() {}

	template <typename NetT>
	void backward(NetT& model) {
		StdTensor<T> dloss = derivative();

		/*
		// SCALING
		if(LOSS_SCALE!=1)
			dloss *= LOSS_SCALE;
		*/

		model.backward(dloss);

		return;
	}

	virtual StdTensor<T> derivative(){
		return StdTensor<T>();
	}

	template <typename CustomType>
	CustomType item() const {
		return CustomType(loss);
	}

	lossT item() const {
		return loss;
	}

protected:
	lossT loss=0;
};

#endif /* LOSS_HPP */

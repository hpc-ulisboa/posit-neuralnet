#ifndef LOSS_HPP
#define LOSS_HPP

// Custom headers
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

enum class Reduction {Mean, Sum};

template <typename BackwardT, typename lossT=float>
class Loss{
public:
	Loss() {}
	virtual ~Loss() {}

	template <typename Model>
	void backward(Model& model) {
		StdTensor<BackwardT> dloss = derivative();

		model.backward(dloss);

		return;
	}

	virtual StdTensor<BackwardT> derivative(){
		return StdTensor<BackwardT>();
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

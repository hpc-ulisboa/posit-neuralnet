#ifndef MIXEDTENSOR_HPP
#define MIXEDTENSOR_HPP

// General headers
#include <vector>

// Custom headers
#include "../tensor/StdTensor.hpp"

class TensorUpdater {
public:
	virtual void update() { }
};

template <typename OptimizerT, typename ForwardT=OptimizerT, typename BackwardT=ForwardT>
class MixedTensor : public TensorUpdater {
public:
	MixedTensor() :
		new_forward(false),
		new_backward(false)
	{ }

	MixedTensor(std::vector<size_t> const& shape) :
		optimizer(shape),
		new_forward(!std::is_same<OptimizerT, ForwardT>::value),
		new_backward(!std::is_same<OptimizerT, BackwardT>::value && !std::is_same<ForwardT, BackwardT>::value)
	{
		if(new_forward) {
			forward = new StdTensor<ForwardT>(shape);
		}
		else {
			forward = (StdTensor<ForwardT>*) &optimizer;
		}

		if(new_backward) {
			backward = new StdTensor<BackwardT>(shape);
		}
		else if (std::is_same<OptimizerT, BackwardT>::value) {
			backward = (StdTensor<BackwardT>*) &optimizer;
		}
		else {
			backward = (StdTensor<BackwardT>*) forward;
		}
	}

	MixedTensor(size_t const size) : MixedTensor(std::initializer_list<size_t>{size}) { }

	~MixedTensor() {
		if(new_forward)
			delete forward;

		if(new_backward)
			delete backward;
	}

	void update() override {
		if (new_forward)
			*forward = optimizer;

		if (new_backward)
			*backward = optimizer;
	}

	StdTensor<OptimizerT>& get_optimizer() { return optimizer; }
	StdTensor<ForwardT>& get_forward() { return *forward; }
	StdTensor<BackwardT>& get_backward() { return *backward; }

private:
	StdTensor<OptimizerT> optimizer;
	StdTensor<ForwardT>* forward;
	StdTensor<BackwardT>* backward;
	bool new_forward;
	bool new_backward;
};

#endif /* MIXEDTENSOR_HPP */

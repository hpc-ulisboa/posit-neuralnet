#ifndef PARAMETER_HPP
#define PARAMETER_HPP

// General headers
#include <vector>

// Custom headers
#include "../tensor/MixedTensor.hpp"
#include "../tensor/StdTensor.hpp"

template <typename T>
struct Parameter {
	Parameter(StdTensor<T>& _weight, StdTensor<T>& _gradient) :
		weight(_weight),
		gradient(_gradient),
		mixed_tensor(nullptr)
	{ }

	template <typename ForwardT, typename BackwardT>
	Parameter(MixedTensor<T, ForwardT, BackwardT>& _weight, StdTensor<T>& _gradient) :
		weight(_weight.get_optimizer()),
		gradient(_gradient),
		mixed_tensor(static_cast<TensorUpdater*>(&_weight))
	{ }

	void update() {
		if(mixed_tensor != nullptr)
			mixed_tensor->update();
	};

	friend std::ostream & operator << (std::ostream& out, const Parameter& parameter){
		out << parameter.weight << std::endl;
		return out;
	}

	StdTensor<T>& weight;
	StdTensor<T>& gradient;
	TensorUpdater* mixed_tensor;
};

/*

template <typename T>
class Parameters {
public:
	Parameters() {}

	void register_parameter(Parameter<T>& p) {
		parameters.push_back(p);
		return;
	}

	typename std::vector<Parameter<T>>::iterator begin() {
		return parameters.begin();
	}

	typename std::vector<Parameter<T>>::iterator end() {
		return parameters.end();
	}

	const Parameter<T>& operator[](size_t i) const {
		return parameters[i];
	}

	Parameter<T>& operator[](size_t i) {
		return parameters[i];
	}

	size_t size() const {
		return parameters.size();
	}	

private:
	std::vector<Parameter<T>> parameters;
};

*/

#endif /* PARAMETER_HPP */

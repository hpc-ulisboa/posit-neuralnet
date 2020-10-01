#ifndef RELU_HPP
#define RELU_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

class ReLU {
public:	
	ReLU() { }

	template <typename T>
	StdTensor<T> forward(StdTensor<T> x) {
		zero.resize(x.size());

		for(size_t i=0, size=x.size(); i<size; i++) {
			if(x[i].isneg() || x[i].iszero()) {
				x[i].setzero();
				zero[i] = true;
			}
			else {
				zero[i] = false;
			}
		}

		return x;
	}

	template <typename T>
	StdTensor<T> backward(StdTensor<T> delta) {
		for(size_t i=0, size=delta.size(); i<size; i++){
			if(zero[i]) {
				delta[i].setzero();
			}
		}

		return delta;
	}

private:
	std::vector<bool> zero;
};

#endif /* RELU_HPP */

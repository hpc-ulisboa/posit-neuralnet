#ifndef RELU_HPP
#define RELU_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename Posit>
class ReLU {
public:	
	ReLU() { }

	StdTensor<Posit> forward(StdTensor<Posit>& x) {
		output = x;

		for(size_t i=0, size=output.size(); i<size; i++) {
			if(output[i].isneg() || output[i].iszero()) {
				output[i].setzero();
			}
		}

		return output;
	}

	StdTensor<Posit> backward(StdTensor<Posit> delta) {
		for(size_t i=0, size=delta.size(); i<size; i++){
			if(output[i].isneg() || output[i].iszero()) {
				delta[i].setzero();
			}
		}

		return delta;
	}

private:
	StdTensor<Posit> output;
};

#endif /* RELU_HPP */

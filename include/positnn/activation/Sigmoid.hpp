#ifndef SIGMOID_HPP
#define SIGMOID_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../tensor/StdTensor.hpp"
#include "../utils/utils.hpp"

// Namespaces
using namespace sw::unum;

template <typename Posit>
class Sigmoid {
public:	
	Sigmoid() { }

	StdTensor<Posit> forward(StdTensor<Posit>& x, bool approximate=true) {
		constexpr size_t es = Posit::es;
		output = StdTensor<Posit>(x.shape());

		for(size_t i=0, size=output.size(); i<size; i++) {
			if(approximate && es==0) {
				output[i] = sigmoid_approx(x[i]);
			}
			else {
				output[i] = 1/(1+exp(-x[i]));
			}
		}

		return output;
	}

	StdTensor<Posit> backward(StdTensor<Posit>& w_delta) {
		StdTensor<Posit> deltaN = derivative();
		deltaN *= w_delta;
		return deltaN;
	}

	StdTensor<Posit> derivative() const {
		// TODO: protect against initialized output
		StdTensor<Posit> dx(output);

		for(Posit& p : dx)
			p *= (1-p);
		
		return dx;
	}

private:
	StdTensor<Posit> output;
};

#endif /* SIGMOID_HPP */

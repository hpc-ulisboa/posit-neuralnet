#ifndef TANH_HPP
#define TANH_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../tensor/StdTensor.hpp"
#include "../utils/utils.hpp"

// Namespaces
using namespace sw::unum;

template <typename ForwardT, typename BackwardT=ForwardT>
class Tanh {
public:	
	Tanh() { }

	StdTensor<ForwardT> forward(StdTensor<ForwardT> const& x, bool approximate=true) {
		StdTensor<ForwardT> y(x.shape());
		
		for(size_t i=0, size=x.size(); i<size; i++) {
			if(approximate && ForwardT::es==0) {
				y[i] = tanh_approx(x[i]);
			}
			else {
				ForwardT plus = exp(x[i]);
				ForwardT minus = exp(-x[i]);
				y[i] = (plus-minus)/(plus+minus);
			}
		}

		output = y;

		return y;
	}

	StdTensor<BackwardT> backward(StdTensor<BackwardT> const& w_delta) {
		StdTensor<BackwardT> deltaN = derivative();
		deltaN *= w_delta;
		return deltaN;
	}

	StdTensor<BackwardT> derivative() const {
		// TODO: protect against initialized output
		StdTensor<BackwardT> dx(output.shape());

		BackwardT pOne(1);

		for(size_t i=0, size=dx.size(); i<size; i++) {
			convert( fma(output[i], -output[i], pOne) ,
					 dx[i] );
		}
		
		return dx;
	}

private:
	StdTensor<BackwardT> output;
};

#endif /* TANH_HPP */

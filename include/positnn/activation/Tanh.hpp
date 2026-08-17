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
				// Case split instead of (e^x - e^-x)/(e^x + e^-x): for large |x| one
				// of the two calls evaluates exp() at a large positive argument, the
				// double fallback returns inf, and inf -> posit is NaR. This is the
				// same overflow that made Sigmoid produce NaR.
				//
				//   x >= 0:  tanh(x) = (1 - e^-2x) / (1 + e^-2x)
				//   x <  0:  tanh(x) = (e^2x - 1) / (e^2x + 1)
				//
				// The argument of exp() is now always <= 0, so it can only underflow
				// and the result saturates at +-1 as it should.
				//
				// As in Sigmoid.hpp, this is a workaround. The real fix belongs in
				// universal's exp(), which should saturate at maxpos instead of
				// producing NaR when its double fallback overflows.
				ForwardT const two(2);
				if(x[i].isneg()) {
					ForwardT const e = exp(two*x[i]);
					y[i] = (e - ForwardT(1)) / (e + ForwardT(1));
				}
				else {
					ForwardT const e = exp(-two*x[i]);
					y[i] = (ForwardT(1) - e) / (ForwardT(1) + e);
				}
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

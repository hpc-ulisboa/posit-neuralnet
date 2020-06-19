#ifndef TANH_HPP
#define TANH_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../tensor/StdTensor.hpp"
#include "../utils/utils.hpp"

// Namespaces
using namespace sw::unum;

template <typename Posit>
class Tanh {
public:	
	Tanh() { }

	StdTensor<Posit> forward(StdTensor<Posit>& x, bool approximate=true) {
		constexpr size_t es = Posit::es;
		output = StdTensor<Posit>(x.shape());

		for(size_t i=0, size=output.size(); i<size; i++) {
			if(approximate && es==0) {
				output[i] = tanh_approx(x[i]);
			}
			else {
				Posit plus = exp(x[i]);
				Posit minus = exp(-x[i]);
				output[i] = (plus-minus)/(plus+minus);
			}
		}

		return output;
	}

	// CHANGE
	StdTensor<Posit> backward(StdTensor<Posit>& w_delta) {
		StdTensor<Posit> deltaN = derivative();
		deltaN *= w_delta;
		return deltaN;
	}

	// CHANGE
	StdTensor<Posit> derivative() const {
		// TODO: protect against initialized output
		StdTensor<Posit> dx(output);

		quire<Posit::nbits, Posit::es, Posit::nbits-1> q;

		for(Posit& p : dx) {
			q = 1;
			q -= quire_mul(p, p);
			convert(q.to_value(), p);
		}
		
		return dx;
	}

private:
	StdTensor<Posit> output;
};

#endif /* TANH_HPP */

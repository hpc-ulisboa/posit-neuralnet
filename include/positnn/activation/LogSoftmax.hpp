#ifndef LOGSOFTMAX_HPP
#define LOGSOFTMAX_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../tensor/StdTensor.hpp"
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

template <typename Posit>
class LogSoftmax {
public:	
	LogSoftmax() { }

	StdTensor<Posit> forward(StdTensor<Posit>& x) {
		//TODO: protect for tensors with dim!=2
		const size_t batch_size = x.shape()[0];
		const size_t sample_size = x.shape()[1];

		exp_x_max = StdTensor<Posit>({batch_size, sample_size});
		sum_exp = StdTensor<Posit>(batch_size);

		StdTensor<Posit> output = StdTensor<Posit>(x);
		Posit max, delta;

		Quire<Posit::nbits, Posit::es> q;

		typename StdTensor<Posit>::iterator const x_begin = x.begin();

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size) {
			max = *std::max_element(x_begin+j, x_begin+j+sample_size);

			q.clear();
			for(size_t k=0; k<sample_size; k++) {
				exp_x_max[j+k] = exp(x[j+k] - max);
				q += exp_x_max[j+k];
			}

			convert(q.to_value(), sum_exp[i]);

			delta = max + log(sum_exp[i]);

			for(size_t k=0; k<sample_size; k++)
				output[j+k] -= delta;
		}

		return output;
	}

	StdTensor<Posit> backward(StdTensor<Posit>& w_delta) {
	//TODO: BACKWARD IS INCORRECT. SHOULD USE JACOBIAN INSTEAD OF HADAMARD PRODUCT
		StdTensor<Posit> deltaN = derivative();
		deltaN *= w_delta;
		return deltaN;
	}

	StdTensor<Posit> derivative() const {
		const size_t batch_size = exp_x_max.shape()[0];
		const size_t sample_size = exp_x_max.shape()[1];

		// TODO: protect against initialized output
		StdTensor<Posit> dx({batch_size, sample_size});

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size) {
			for(size_t k=0; k<sample_size; k++) {
				dx[j+k] = (sum_exp[i] - exp_x_max[j+k])/sum_exp[i];
			}
		}
		
		return dx;
	}

private:
	StdTensor<Posit> sum_exp;
	StdTensor<Posit> exp_x_max;
};

#endif /* LOGSOFTMAX_HPP */

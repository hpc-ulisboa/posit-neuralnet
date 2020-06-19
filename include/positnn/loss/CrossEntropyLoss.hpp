#ifndef CROSSENTROPYLOSS_HPP
#define CROSSENTROPYLOSS_HPP

// Custom headers
#include "Loss.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename Posit, typename targetT, typename lossT=float>
class cross_entropy_loss : public Loss<Posit, lossT>{
public:
	cross_entropy_loss() { }

	cross_entropy_loss(StdTensor<Posit>& x, StdTensor<targetT>& _target, Reduction reduction=Reduction::Mean) :
		exp_x_max(x.shape()),
		sum(x.shape()[0]),
		target(_target)
	{
		// TODO: protect if target is not integer
		const size_t batch_size = x.shape()[0];
		const size_t sample_size = x.shape()[1];

		Posit max, log_softmax;
		quire<Posit::nbits, Posit::es, Posit::nbits-1> q;
		typename StdTensor<Posit>::iterator const x_begin = x.begin();

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size) {
			max = *std::max_element(x_begin+j, x_begin+j+sample_size);

			q = 0;
			for(size_t k=0; k<sample_size; k++) {
				exp_x_max[j+k] = exp(x[j+k] - max);
				q += exp_x_max[j+k];
			}
			convert(q.to_value(), sum[i]);

			log_softmax = x[j+target[i]] - max - log(sum[i]);

			this->loss -= lossT( log_softmax );
		}

		if(reduction == Reduction::Mean)
			this->loss /= batch_size;
	}	

	StdTensor<Posit> derivative() override {
		size_t const batch_size = exp_x_max.shape()[0];
		size_t const sample_size = exp_x_max.shape()[1];

		StdTensor<Posit> dloss(exp_x_max);

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size){
			// Calculate softmax and subtract 1 to the target class
			const size_t index = target[i];
			for(size_t k=0; k<sample_size; k++) {
				if(k==index)
					dloss[j+k] -= sum[i];
				dloss[j+k] /= sum[i];
			}
		}

		return dloss;
	}

private:
	StdTensor<Posit> exp_x_max;
	StdTensor<Posit> sum;
	StdTensor<targetT> target;
};

#endif /* CROSSENTROPYLOSS_HPP */

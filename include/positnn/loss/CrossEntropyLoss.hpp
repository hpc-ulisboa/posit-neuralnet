#ifndef CROSSENTROPYLOSS_HPP
#define CROSSENTROPYLOSS_HPP

// Custom headers
#include "Loss.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <class Posit, class outputT, typename targetT, typename lossT=float>
class cross_entropy_loss : public Loss<Posit, lossT>{
public:
	cross_entropy_loss() { }

	cross_entropy_loss(StdTensor<outputT>& output, StdTensor<targetT>& _target, Reduction reduction=Reduction::Mean) :
		exp_x_max(output.shape()),
		sum(output.shape()[0]),
		target(_target)
	{
		// TODO: protect if target is not integer
		const size_t batch_size = output.shape()[0];
		const size_t sample_size = output.shape()[1];

		StdTensor<Posit> x(output);
		Posit max, log_softmax;
		quire<Posit::nbits, Posit::es, Posit::nbits-1> q;
		typename StdTensor<outputT>::iterator const output_begin = output.begin();

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size) {
			max = *std::max_element(output_begin+j, output_begin+j+sample_size);

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

		// Previously used this (divide last)
		/*
		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size){
			// Calculate softmax and subtract 1 to the target class
			const size_t index = target[i];
			for(size_t k=0; k<sample_size; k++) {
				if(k==index) {
					dloss[j+k] -= sum[i];
				}

				dloss[j+k] /= sum[i];
			}
		}
		*/

		// Now I use this and seems to give better results (divide first)
		/*
		Posit pOne(1);

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size){
			// Calculate softmax and subtract 1 to the target class
			const size_t index = target[i];
			for(size_t k=0; k<sample_size; k++) {
				dloss[j+k] /= sum[i];
				if(k==index)
					dloss[j+k] -= pOne;
			}
		}
		*/

		// This has even better results (fma)
		Posit pMinusOne(-1);
		value<1 + 2 * (Posit::nbits - Posit::es)> result;

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size){
			// Calculate softmax and subtract 1 to the target class
			const size_t index = target[i];
			Posit den = sum[i].reciprocate();

			for(size_t k=0; k<sample_size; k++) {
				if(k==index) {
					result = fma(dloss[j+k], den, pMinusOne);
					convert(result, dloss[j+k]);
				}
				else {
					dloss[j+k] /= sum[i];
				}
			}
		}

		// This has the best results (fam)
		/*
		value<2 * (Posit::nbits + 3 - Posit::es)> result;

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size){
			// Calculate softmax and subtract 1 to the target class
			const size_t index = target[i];
			Posit sub(-sum[i]);
			//Posit den(1/sum[i]);
			Posit den = sum[i].reciprocate();

			for(size_t k=0; k<sample_size; k++) {
				if(k==index) {
					result = fam_corrected(dloss[j+k], sub, den);
					convert(result, dloss[j+k]);
				}
				else {
					dloss[j+k] /= sum[i];
				}
			}
		}
		*/

		// This should have better results than fma (fma improved)
		/*
		Posit pMinusOne(-1);
		value<1 + 2 * (Posit::nbits - Posit::es)> result;

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size){
			// Calculate softmax and subtract 1 to the target class
			const size_t index = target[i];

			for(size_t k=0; k<sample_size; k++) {
				if(k==index) {
					if(dloss[j+k]==sum[i]) {
						dloss[j+k].setzero();
					}
					else {
						Posit den = sum[i].reciprocate();
						result = fma(dloss[j+k], den, pMinusOne);
						convert(result, dloss[j+k]);
					}
				}
				else {
					dloss[j+k] /= sum[i];
				}
			}
		}
		*/

		return dloss;
	}

private:
	StdTensor<Posit> exp_x_max;
	StdTensor<Posit> sum;
	StdTensor<targetT> target;
};

#endif /* CROSSENTROPYLOSS_HPP */

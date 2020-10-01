#ifndef CROSSENTROPYLOSS_HPP
#define CROSSENTROPYLOSS_HPP

// Custom headers
#include "Loss.hpp"
#include "../tensor/StdTensor.hpp"
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

template <class ForwardT, class BackwardT=ForwardT, class TargetT=unsigned short int, typename lossT=float>
class cross_entropy_loss : public Loss<BackwardT, lossT>{
public:
	cross_entropy_loss() { }

	cross_entropy_loss(StdTensor<ForwardT> const& output, StdTensor<TargetT> const& _target, Reduction reduction=Reduction::Mean) :
		exp_x_max(output.shape()),
		sum(output.shape()[0]),
		target(_target)
	{
		// TODO: protect if target is not integer
		const size_t batch_size = output.shape()[0];
		const size_t sample_size = output.shape()[1];

		typename StdTensor<ForwardT>::const_iterator const output_begin = output.begin();
		Quire<ForwardT::nbits, ForwardT::es> q;
		ForwardT sum_forward;

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size) {
			ForwardT const& max = *std::max_element(output_begin+j, output_begin+j+sample_size);

			q.clear();
			for(size_t k=0; k<sample_size; k++) {
				ForwardT const exp_x_max_forward = exp(output[j+k] - max);
				q += exp_x_max_forward;

				// Copy to be used in backward
				exp_x_max[j+k] = exp_x_max_forward;
			}
			convert(q.to_value(), sum_forward);

			// Copy to be used in backward
			sum[i] = sum_forward;

			ForwardT const log_softmax = output[j+target[i]] - max - log(sum_forward);

			this->loss -= lossT( log_softmax );
		}

		if(reduction == Reduction::Mean)
			this->loss /= batch_size;
	}	

	StdTensor<BackwardT> derivative() override {
		size_t const batch_size = exp_x_max.shape()[0];
		size_t const sample_size = exp_x_max.shape()[1];


		// Previously used this (divide last)
		/*
		StdTensor<BackwardT> dloss(exp_x_max);

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
		StdTensor<BackwardT> dloss(exp_x_max);
		BackwardT pOne(1);

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
		/*
		StdTensor<BackwardT> dloss(exp_x_max.shape());
		BackwardT pMinusOne(-1);
		value<1 + 2 * (BackwardT::nbits - BackwardT::es)> result;

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size){
			// Calculate softmax and subtract 1 to the target class
			const size_t index = target[i];
			BackwardT den = sum[i].reciprocate();

			for(size_t k=0; k<sample_size; k++) {
				if(k==index) {
					result = fma(exp_x_max[j+k], den, pMinusOne);
					convert(result, dloss[j+k]);
				}
				else {
					dloss[j+k] = exp_x_max[j+k]/sum[i];
				}
			}
		}
		*/

		// This has the best results (fam)
		StdTensor<BackwardT> dloss(exp_x_max.shape());
		value<2 * (BackwardT::nbits + 3 - BackwardT::es)> result;

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size){
			// Calculate softmax and subtract 1 to the target class
			const size_t index = target[i];
			BackwardT sub(-sum[i]);
			//BackwardT den(1/sum[i]);
			BackwardT den = sum[i].reciprocate();

			for(size_t k=0; k<sample_size; k++) {
				if(k==index) {
					result = fam_corrected(exp_x_max[j+k], sub, den);
					convert(result, dloss[j+k]);
				}
				else {
					dloss[j+k] = exp_x_max[j+k]/sum[i];
				}
			}
		}

		// This should have better results than fma (fma improved)
		/*
		StdTensor<BackwardT> dloss(exp_x_max.shape());
		BackwardT pMinusOne(-1);
		value<1 + 2 * (BackwardT::nbits - BackwardT::es)> result;

		for(size_t i=0, j=0; i<batch_size; i++, j+=sample_size){
			// Calculate softmax and subtract 1 to the target class
			const size_t index = target[i];

			for(size_t k=0; k<sample_size; k++) {
				if(k==index) {
					if(exp_x_max[j+k]==sum[i]) {
						dloss[j+k].setzero();
					}
					else {
						BackwardT den = sum[i].reciprocate();
						result = fma(exp_x_max[j+k], den, pMinusOne);
						convert(result, dloss[j+k]);
					}
				}
				else {
					dloss[j+k] = exp_x_max[j+k]/sum[i];
				}
			}
		}
		*/

		return dloss;
	}

private:
	StdTensor<BackwardT> exp_x_max;
	StdTensor<BackwardT> sum;
	StdTensor<TargetT> target;
};

#endif /* CROSSENTROPYLOSS_HPP */

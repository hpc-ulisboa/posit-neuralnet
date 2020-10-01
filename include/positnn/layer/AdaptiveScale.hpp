#ifndef ADAPTIVESCALE_HPP
#define ADAPTIVESCALE_HPP

// General headers
#include <cmath>
#include <universal/posit/posit>
#include <vector>

// Custom headers
//#include "Layer.hpp"
#include "../layer/Parameter.hpp"
#include "../tensor/StdTensor.hpp"
#include "../tensor/stats.hpp"

// Namespaces
using namespace sw::unum;

enum class AdaptiveScaleMode {Default, Normalize, Half};

template <typename OptimizerT, typename FactorsT=OptimizerT, typename StatsT=FactorsT>
class AdaptiveScale {
//class AdaptiveScale : public Layer<Posit> {
public:
	AdaptiveScale(size_t const _nlayers, std::vector<Parameter<OptimizerT>>& _parameters, AdaptiveScaleMode _mode=AdaptiveScaleMode::Default, StatsT _momentum=0.1, bool _use_pow2=false) :
		n(_nlayers, 1),
		std(_nlayers, StatsT(1)),
		running_std(_nlayers, StatsT(1)),
		scale(_nlayers, FactorsT(1)),
		acc_scale(_nlayers, FactorsT(1)),
		nlayers(_nlayers),
		mode(_mode),
		momentum(_momentum),
		use_pow2(_use_pow2),
		parameters(_parameters),
		nparameters(_nlayers)
	{ 
		state = disabled;
	}

	enum State {enabled, disabled, setuping};

	// Backward scale but doesn't correct gradients
	template <typename BackwardT>
	StdTensor<BackwardT> backward(size_t const i, StdTensor<BackwardT> const& x, size_t npar, bool linear=true) {
		if(state==setuping) {
			n[i] = x.size();
			std[i] = estimate_std<StatsT>(i, x, linear);
			running_std[i] = running_std[i]*(StatsT(1)-momentum) + std[i]*momentum;

			BackwardT u(0);

			switch(mode) {
				case AdaptiveScaleMode::Default:
					for(size_t i=0, size=1<<(BackwardT::es+1); i<size; i++) {
						u++;		// first posit with fraction
					}
					scale[i] = running_std[i]*(StatsT(1.25331447e-3)*StatsT(u));
					break;

				case AdaptiveScaleMode::Normalize:
					scale[i] = running_std[i];
					break;

				case AdaptiveScaleMode::Half:
					scale[i] = running_std[i]*0.6745;
					break;
			}

			if(use_pow2)
				scale[i] = round_pow2(scale[i]);
			
			if(i+1==nlayers)
				acc_scale[i] = scale[i];
			else
				acc_scale[i] = scale[i] * acc_scale[i+1];

			nparameters[i] = npar;
		}

		if(state==enabled || state==setuping) {
			if(!scale[i].isone() && !scale[i].iszero())
				return x / scale[i];
		}
		
		return x;
	}

	void scale_gradients() {
		size_t begin = 0;
		for(size_t i=0; i<nlayers; i++) {
			for(size_t j=begin, end=begin+nparameters[i]; j<end; j++) {
				if(!acc_scale[i].isone() && !acc_scale[i].iszero()) {
					parameters[j].gradient *= acc_scale[i];
				}
			}
			begin += nparameters[i];
		}
	}

	void enable() {
		state = enabled;
	}

	void disable() {
		state = disabled;
	}

	void setup() {
		state = setuping;
	}

	std::vector<size_t>& sizes() {
		return n;
	}

	std::vector<FactorsT>& scale_factors() {
		return scale;
	}

	std::vector<FactorsT>& acc_scale_factors() {
		return acc_scale;
	}

	std::vector<StatsT>& stddev() {
		return std;
	}

	std::vector<StatsT>& running_stddev() {
		return running_std;
	}

	void print_stats() {
		std::cout << " n: " << n << std::endl;
		std::cout << " stdev: " << std << std::endl;
		std::cout << " running: " << running_std << std::endl;
		std::cout << " scale: " << scale << std::endl;
		std::cout << " acc_scale: " << acc_scale << std::endl;
	}

private:
	template <typename BackwardT>
	StatsT estimate_std(size_t const i, StdTensor<BackwardT> const& x, bool linear) {
		size_t idx = std::accumulate(nparameters.begin(), nparameters.begin()+i, static_cast<size_t>(0));
		StatsT var1 = calculate_var<StatsT>(x);
		StatsT std;

		if(linear) {
			StatsT mean1 = calculate_mean<StatsT>(x);
			StatsT mean2 = calculate_mean<StatsT>(parameters[idx].weight);
			StatsT var2 = calculate_var<StatsT>(parameters[idx].weight);

			mean1 *= mean1;
			mean2 *= mean2;

			std = sqrt((var1+mean1)*(var2+mean2)-(mean1*mean2));
			std *= std::sqrt(parameters[idx].weight.shape()[1]);
		}
		else {
			StdTensor<StatsT> weight = parameters[idx].weight;

			// TODO: implement with quires?
			weight *= weight;	// square
			weight *= var1;

			std = sqrt(weight.sum()/weight.shape()[1]);
		}

		return std;
	}

	std::vector<size_t> n;
	std::vector<StatsT> std;
	std::vector<StatsT> running_std;
	std::vector<FactorsT> scale;
	std::vector<FactorsT> acc_scale;

	size_t const nlayers;
	State state;
	AdaptiveScaleMode mode;
	StatsT momentum;
	bool use_pow2;
	std::vector<Parameter<OptimizerT>> parameters;
	std::vector<size_t> nparameters;
};

#endif /* ADAPTIVESCALE_HPP */

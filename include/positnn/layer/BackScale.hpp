#ifndef BACKSCALE_HPP
#define BACKSCALE_HPP

// General headers
#include <universal/posit/posit>
#include <vector>

// Custom headers
//#include "Layer.hpp"
#include "../layer/Parameter.hpp"
#include "../tensor/StdTensor.hpp"
#include "../tensor/stats.hpp"
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

enum class BackScaleMode {Loss, LogLoss, MultiLog, Mix, Before, After, Half};

template <typename OptimizerT, typename BackwardT=OptimizerT, typename FactorsT=OptimizerT, typename StatsT=FactorsT>
class BackScale {
//class BackScale : public Layer<Posit> {
public:
	BackScale(size_t const _nlayers, BackScaleMode _mode=BackScaleMode::Loss, StatsT _momentum=0.9, bool _use_pow2=false) :
		n(_nlayers, 1),
		std(_nlayers, StatsT(1)),
		running_std(_nlayers, StatsT(1)),
		scale(_nlayers, FactorsT(1)),
		acc_scale(_nlayers, FactorsT(1)),
		nlayers(_nlayers),
		mode(_mode),
		momentum(_momentum),
		use_pow2(_use_pow2)
	{ 
		state = disabled;
		indices.reserve(nlayers);
	}

	enum State {enabled, disabled, setuping, setuping_with_scale};

	// Backward scale but doesn't correct gradients
	StdTensor<BackwardT> backward(size_t const i, StdTensor<BackwardT> x) {
		if(state==setuping || state==setuping_with_scale) {
			n[i] = x.size();
			std[i] = calculate_std<StatsT>(x);

			if(state==setuping_with_scale && i+1<nlayers && !acc_scale[i+1].isone())
				std[i] *= acc_scale[i+1];

			running_std[i] = running_std[i]*momentum + std[i]*(StatsT(1)-momentum);

			calculate_factors();
		}

		if((state==enabled || state==setuping_with_scale) && !scale[i].isone() && !scale[i].iszero())
			x /= scale[i];

		return x;
	}

	// Backward scale and correct gradients
	StdTensor<BackwardT> backward(size_t const i, StdTensor<BackwardT> x, std::vector<Parameter<OptimizerT>>& parameters) {
		// Scale gradients
		if((state==enabled || state==setuping_with_scale) && i+1<nlayers && !acc_scale[i+1].isone() && !scale[i+1].iszero()){
			for(Parameter<OptimizerT>& p : parameters) {
				p.gradient *= acc_scale[i+1];
			}
		}

		return backward(i, x);
	}

	void register_indices(std::vector<size_t> _indices) {
		indices = _indices;
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

	void setup_with_scale() {
		state = setuping_with_scale;
	}

	void calculate_factors() {
		//if(state!=setuping && state!=setuping_with_scale)
		//	return;

		switch(mode) {
			case BackScaleMode::Loss:
				calculate_factors_loss();
				break;
			case BackScaleMode::LogLoss:
				calculate_factors_logloss();
				break;
			case BackScaleMode::MultiLog:
				calculate_factors_multilog();
				break;
			case BackScaleMode::Mix:
				calculate_factors_mix();
				break;
			case BackScaleMode::Before:
				calculate_factors_before();
				break;
			case BackScaleMode::After:
				calculate_factors_after();
				break;
			case BackScaleMode::Half:
				calculate_factors_half();
				break;
			default:
				std::cerr << "Undefined BackScaleMode" << std::endl;
				return;
		}

		if(use_pow2) {
			for(BackwardT& p : scale) {
				p = round_pow2(p);
			}
		}

		FactorsT prod(1);
		for(size_t i=nlayers; i-->0; ){
			prod *= scale[i];
			acc_scale[i] = prod;
		}
	}

	std::vector<size_t>& sizes() {
		return n;
	}

	std::vector<BackwardT>& scale_factors() {
		return scale;
	}

	std::vector<OptimizerT>& acc_scale_factors() {
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

	typedef FactorsT factors_type;
	
private:
	// Optimize for before and after
	void calculate_factors_mix() {
		std::vector<FactorsT> s;
		std::vector<FactorsT> c;
		
		s.reserve(nlayers);
		c.reserve(nlayers);

		FactorsT pZero(0);

		for(size_t i=0; i<nlayers; i++) {
			if(running_std[i].iszero())
				s.push_back(pZero);
			else
				s.push_back(log10(running_std[i]));
		}

		//std::cout << "s: " << s << std::endl;

		// First C
		c.push_back(0);

		// Middle C's
		for(size_t i=1; i<nlayers-1; i++) {
			c.push_back(	((n[i-1]*n[i])*(s[i-1]-s[i]) + 
							(n[i-1]*n[i+1])*(s[i-1]-s[i+1]) + 
							(n[i]*n[i+1])*(s[i]-s[i+1])) / 
							((n[i-1]+n[i])*(n[i]+n[i+1]))	);
		}

		// Last C
		c.push_back((n[nlayers-2]*s[nlayers-2] + n[nlayers-1]*s[nlayers-1]) /
					(n[nlayers-2] + n[nlayers-1]));

		//std::cout << "c: " << c << std::endl;

		// Convert C to scale factors
		scale[0] = 1;
		FactorsT const pTen(10);
		for(size_t i=1; i<nlayers; i++) {
			scale[i] = pow(pTen, c[i]);
		}

		return;
	}

	// Loss scale: weighted average of std
	void calculate_factors_loss() {
		//std::cout << "n: " << n << std::endl;
		//std::cout << "std: " << std << std::endl;

		BackwardT const pOne(1);
		for(size_t i=0, size=nlayers-1; i<size; i++) {
			scale[i] = pOne;
		}

		size_t sum_n = 0;
		FactorsT aux;
		Quire<FactorsT::nbits, FactorsT::es> sum;

		sum.clear();
		for(size_t i=0; i<nlayers; i++) {
			sum_n += n[i];
			sum += Quire_mul(FactorsT(n[i]), FactorsT(running_std[i]));
		}

		convert(sum.to_value(), aux);
		if(aux.iszero())
			aux = 1;
		else
			aux /= sum_n;

		scale.back() = aux;

		return;
	}

	// Log loss scale: weighted average of log(std)
	void calculate_factors_logloss() {
		size_t sum_n = 0;
		Quire<FactorsT::nbits, FactorsT::es> sum;
		sum.clear();

		for(size_t i=0; i<nlayers; i++) {
			sum_n += n[i];

			if(running_std[i].ispos() && !running_std[i].iszero()) {
				FactorsT log_std = log10(running_std[i]);
				sum += Quire_mul(FactorsT(n[i]), log_std);
			}
		}

		FactorsT aux;
		convert(sum.to_value(), aux);

		if(aux.iszero())
			aux = 1;
		else {
			aux /= sum_n;
			aux = pow(FactorsT(10), aux);
		}

		BackwardT const pOne(1);
		for(size_t i=0, size=nlayers-1; i<size; i++) {
			scale[i] = pOne;
		}

		scale.back() = aux;

		return;
	}
	
	// Multi log scale: weighted average of log(std) applied to multiple layers
	void calculate_factors_multilog() {
		FactorsT const pTen(10);

		Quire<FactorsT::nbits, FactorsT::es> sum;
		FactorsT log_acc_factor(0);

		for(size_t i=0, n_indices=indices.size(); i<n_indices; i++) {
			size_t const current = indices[i];
			size_t const next = (i+1<n_indices) ? indices[i+1] : 0;

			size_t sum_n = 0;
			sum.clear();

			for(size_t j=current; j-->next; ) {
				sum_n += n[j];

				if(running_std[j].ispos() && !running_std[j].iszero()) {
					FactorsT log_std = log10(running_std[j]);
					sum += Quire_mul(FactorsT(n[j]), log_std);
				}
			}

			FactorsT factor;
			convert(sum.to_value(), factor);

			if(factor.iszero()) {
				factor = 1;
			}
			else {
				factor /= sum_n;

				factor -= log_acc_factor;
				log_acc_factor += factor;

				factor = pow(pTen, factor);
			}

			// Fill scale factors from current to next
			scale[current] = factor;

			for(size_t j=current; j-->next; ) {
				scale[j] = 1;
			}
		}

		return;
	}

	
	// Optimize for before
	void calculate_factors_before() {
		std::vector<FactorsT> s;
		std::vector<FactorsT> c;
		
		s.reserve(nlayers);
		c.reserve(nlayers);

		FactorsT const pZero(0);

		for(size_t i=0; i<nlayers; i++) {
			if(running_std[i].iszero())
				s.push_back(pZero);
			else
				s.push_back(log10(running_std[i]));
		}

		//std::cout << "s: " << s << std::endl;

		// First C
		c.push_back(0);

		// Middle C's
		for(size_t i=1, size=nlayers-1; i<size; i++) {
			c.push_back(s[i-1]-s[i]);
		}

		// Last C
		c.push_back(s[nlayers-2]);

		//std::cout << "c: " << c << std::endl;

		// Convert C to scale factors
		scale[0] = 1;
		FactorsT const pTen(10);
		for(size_t i=1; i<nlayers; i++) {
			scale[i] = pow(pTen, c[i]);
		}

		return;
	}
	
	// Optimize for after
	void calculate_factors_after() {
		std::vector<FactorsT> s;
		std::vector<FactorsT> c;
		
		s.reserve(nlayers);
		c.reserve(nlayers);

		FactorsT const pZero(0);

		for(size_t i=0; i<nlayers; i++) {
			if(running_std[i].iszero())
				s.push_back(pZero);
			else
				s.push_back(log10(running_std[i]));
		}

		//std::cout << "s: " << s << std::endl;

		// First C
		c.push_back(0);

		// Middle C's
		for(size_t i=1; i<nlayers-1; i++) {
			c.push_back(s[i]-s[i+1]);
		}

		// Last C
		c.push_back(s[nlayers-1]);

		//std::cout << "c: " << c << std::endl;

		// Convert C to scale factors
		scale[0] = 1;
		FactorsT const pTen(10);
		for(size_t i=1; i<nlayers; i++) {
			scale[i] = pow(pTen, c[i]);
		}

		return;
	}
	
	// Optimize for half
	void calculate_factors_half() {
		StatsT ratio(0.6745);
		StatsT acc(1);

		for(size_t i=nlayers; i-->1;) {
			scale[i] = (running_std[i-1]*ratio)/acc;

			if(use_pow2)
				scale[i] = round_pow2(scale[i]);

			acc *= scale[i];
		}
		scale[0] = 1;

		return;
	}


	std::vector<size_t> n;
	std::vector<StatsT> std;
	std::vector<StatsT> running_std;
	std::vector<BackwardT> scale;
	std::vector<OptimizerT> acc_scale;

	size_t const nlayers;
	State state;
	BackScaleMode mode;
	StatsT momentum;
	bool use_pow2;
	std::vector<size_t> indices;
};

template <typename Model, typename Loss, class BackScaleLayer>
void setup_back_scale(Model& model, Loss& loss, BackScaleLayer& bs) {
	//bs.setup();
	bs.setup_with_scale();

	loss.backward(model);	

	std::cout << " n: " << bs.sizes() << std::endl;
	std::cout << " stdev: " << bs.stddev() << std::endl;
	std::cout << " running: " << bs.running_stddev() << std::endl;

	bs.calculate_factors();

	bs.enable();

	std::cout << " scale: " << bs.scale_factors() << std::endl;
	std::cout << " acc_scale: " << bs.acc_scale_factors() << std::endl;

	return;
}

#endif /* BACKSCALE_HPP */

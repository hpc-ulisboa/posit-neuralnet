#ifndef BACKSCALE_HPP
#define BACKSCALE_HPP

// General headers
#include <universal/posit/posit>
#include <vector>

// Custom headers
#include "Layer.hpp"
#include "../tensor/StdTensor.hpp"
#include "../tensor/stats.hpp"

// Namespaces
using namespace sw::unum;

enum class BackScaleMode {Loss, Mix, Before, After};

template <typename Posit>
class BackScale : Layer<Posit> {
public:
	BackScale(size_t const nlayers, BackScaleMode _mode=BackScaleMode::Mix, bool _use_pow2=false) :
		n(nlayers, 1),
		var(nlayers, Posit(1)),
		scale(nlayers, Posit(1)),
		acc_scale(nlayers, Posit(1)),
		mode(_mode),
		use_pow2(_use_pow2)
	{ 
		state = disabled;
	}

	enum State {enabled, disabled, setuping};

	template <typename T>
	StdTensor<T> forward(StdTensor<T>& x) {
		return x;
	}

	template <typename T>
	StdTensor<T> backward(size_t const i, StdTensor<T> x) {
		if(state == enabled) {
			if(!scale[i].isone() && !scale[i].iszero())
				x /= scale[i];
		}
		else if(state == setuping) {
			n[i] = x.size();
			var[i] = calculate_var<T, Posit>(x);
		}

		return x;
	}

	template <typename T>
	StdTensor<T> backward(size_t const i, StdTensor<T> x, std::vector<Parameter<T>>& parameters) {
		if(state == enabled) {
			if(!scale[i].isone() && !scale[i].iszero()) {
				x /= scale[i];
			}

			if(!acc_scale[i+1].isone() && !scale[i].iszero()){
				for(Parameter<T>& p : parameters) {
					p.gradient *= acc_scale[i+1];
				}
			}
		}
		else if(state == setuping) {
			n[i] = x.size();
			var[i] = calculate_var<T, Posit>(x);
		}

		return x;
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

	template <typename Pow2Posit=Posit>
	void calculate_factors() {
		switch(mode) {
			case BackScaleMode::Loss:
				calculate_factors_loss();
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
			default:
				std::cerr << "Undefined BackScaleMode" << std::endl;
				return;
		}

		if(use_pow2) {
			factors_pow2<Pow2Posit>();
		}
	}

	std::vector<Posit>& scale_factors() {
		return scale;
	}

	std::vector<Posit>& acc_scale_factors() {
		return acc_scale;
	}

	std::vector<Posit> stddev() {
		std::vector<Posit> aux(var.begin(), var.end());
		for(Posit& elem : aux)
			elem = sqrt(elem);
		return aux;
	}
	
private:
	// Optimize for before and after
	void calculate_factors_mix() {
		size_t const size = n.size();

		std::vector<Posit> s;
		std::vector<Posit> c;
		
		s.reserve(size);
		c.reserve(size);

		//std::cout << "var: " << var << std::endl;

		Posit pZero(0);

		for(size_t i=0; i<size; i++) {
			if(var[i].iszero())
				s.push_back(pZero);
			else
				s.push_back(0.5*log10(var[i]));	// calculate log(std)
		}

		//std::cout << "s: " << s << std::endl;

		// First C
		c.push_back(0);

		// Middle C's
		for(size_t i=1; i<size-1; i++) {
			c.push_back(	((n[i-1]*n[i])*(s[i-1]-s[i]) + 
							(n[i-1]*n[i+1])*(s[i-1]-s[i+1]) + 
							(n[i]*n[i+1])*(s[i]-s[i+1])) / 
							((n[i-1]+n[i])*(n[i]+n[i+1]))	);
		}

		// Last C
		c.push_back((n[size-2]*s[size-2] + n[size-1]*s[size-1]) /
					(n[size-2] + n[size-1]));

		//std::cout << "c: " << c << std::endl;

		// Convert C to scale factors
		scale[0] = 1;
		Posit pTen(10);
		for(size_t i=1; i<size; i++) {
			scale[i] = pow(pTen, c[i]);
		}

		Posit sum = 0;
		for(size_t i=n.size(); i-->0;) {
			sum += c[i];
			acc_scale[i] = pow(pTen, sum);
		}

		return;
	}

	// Loss scale: weighted average of std
	void calculate_factors_loss() {
		size_t const size = n.size();

		std::vector<Posit> std;
		std.reserve(size);

		//std::cout << "n: " << n << std::endl;
		//std::cout << "var: " << var << std::endl;

		for(size_t i=0; i<size; i++) {
			std.push_back(sqrt(var[i]));	// calculate std
		}

		std::cout << "std: " << std << std::endl;

		Posit pOne(1);
		for(size_t i=0, size=scale.size()-1; i<size; i++) {
			scale[i] = pOne;
		}

		size_t sum_n = 0;
		Posit aux;
		quire<Posit::nbits, Posit::es, Posit::nbits-1> sum;

		sum.reset();
		for(size_t i=0, size=n.size(); i<size; i++) {
			sum_n += n[i];
			sum += quire_mul(Posit(n[i]), std[i]);
		}
		convert(sum.to_value(), aux);
		if(aux.iszero())
			aux = 1;
		else
			aux /= sum_n;

		scale.back() = aux;
		for(Posit& p : acc_scale)
			p = aux;

		return;
	}
	
	// Optimize for before
	void calculate_factors_before() {
		size_t const size = n.size();

		std::vector<Posit> s;
		std::vector<Posit> c;
		
		s.reserve(size);
		c.reserve(size);

		//std::cout << "var: " << var << std::endl;

		Posit pZero(0);

		for(size_t i=0; i<size; i++) {
			if(var[i].iszero())
				s.push_back(pZero);
			else
				s.push_back(0.5*log10(var[i]));	// calculate log(std)
		}

		//std::cout << "s: " << s << std::endl;

		// First C
		c.push_back(0);

		// Middle C's
		for(size_t i=1; i<size-1; i++) {
			c.push_back(s[i-1]-s[i]);
		}

		// Last C
		c.push_back(s[size-2]);

		//std::cout << "c: " << c << std::endl;

		// Convert C to scale factors
		scale[0] = 1;
		Posit pTen(10);
		for(size_t i=1; i<size; i++) {
			scale[i] = pow(pTen, c[i]);
		}

		Posit sum = 0;
		for(size_t i=n.size(); i-->0;) {
			sum += c[i];
			acc_scale[i] = pow(pTen, sum);
		}

		return;
	}
	
	// Optimize for after
	void calculate_factors_after() {
		size_t const size = n.size();

		std::vector<Posit> s;
		std::vector<Posit> c;
		
		s.reserve(size);
		c.reserve(size);

		//std::cout << "var: " << var << std::endl;

		Posit pZero(0);

		for(size_t i=0; i<size; i++) {
			if(var[i].iszero())
				s.push_back(pZero);
			else
				s.push_back(0.5*log10(var[i]));	// calculate log(std)
		}

		//std::cout << "s: " << s << std::endl;

		// First C
		c.push_back(0);

		// Middle C's
		for(size_t i=1; i<size-1; i++) {
			c.push_back(s[i]-s[i+1]);
		}

		// Last C
		c.push_back(s[size-1]);

		//std::cout << "c: " << c << std::endl;

		// Convert C to scale factors
		scale[0] = 1;
		Posit pTen(10);
		for(size_t i=1; i<size; i++) {
			scale[i] = pow(pTen, c[i]);
		}

		Posit sum = 0;
		for(size_t i=n.size(); i-->0;) {
			sum += c[i];
			acc_scale[i] = pow(pTen, sum);
		}

		return;
	}

	template<typename Pow2Posit>
	void factors_pow2(){
		Pow2Posit aux;

		for(Posit& p : scale) {
			aux = p;
			p = round_pow2(aux);
		}

		for(Posit& p : acc_scale) {
			aux = p;
			p = round_pow2(aux);
		}
	}

	std::vector<size_t> n;
	std::vector<Posit> var;
	std::vector<Posit> scale;
	std::vector<Posit> acc_scale;

	State state;
	BackScaleMode mode;
	bool use_pow2;
};

template <typename ModelPosit, typename Model, typename Loss, typename Posit>
void setup_back_scale(Model& model, Loss& loss, BackScale<Posit>& bs) {
	bs.setup();

	loss.backward(model);	

	bs.template calculate_factors<ModelPosit>();

	bs.enable();

	std::cout << "stdev: " << bs.stddev() << std::endl;
	std::cout << "scale: " << bs.scale_factors() << std::endl;
	std::cout << "acc_scale: " << bs.acc_scale_factors() << std::endl;

	return;
}

#endif /* BACKSCALE_HPP */

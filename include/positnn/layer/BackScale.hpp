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

template <typename Posit>
class BackScale : Layer<Posit> {
public:
	BackScale(size_t const nlayers) :
		n(nlayers, 1),
		var(nlayers, Posit(1)),
		scale(nlayers, Posit(1)),
		acc_scale(nlayers, Posit(1))
	{ }

	enum Mode {enabled, disabled, setuping};

	template <typename T>
	StdTensor<T> forward(StdTensor<T>& x) {
		return x;
	}

	template <typename T>
	StdTensor<T> backward(size_t const i, StdTensor<T> x) {
		if(mode == enabled) {
			x /= scale[i];
		}
		else if(mode == setuping) {
			n[i] = x.size();
			var[i] = calculate_var<T, Posit>(x);
		}

		return x;
	}

	template <typename T>
	StdTensor<T> backward(size_t const i, StdTensor<T> x, std::vector<Parameter<T>>& parameters) {
		if(mode == enabled) {
			if(!scale[i].isone())
				x /= scale[i];

			if(!acc_scale[i+1].isone()){
				for(Parameter<T>& p : parameters)
					p.gradient *= acc_scale[i+1];
			}
		}
		else if(mode == setuping) {
			n[i] = x.size();
			var[i] = calculate_var<T, Posit>(x);
		}

		return x;
	}

	void enable() {
		mode = enabled;
	}

	void disable() {
		mode = disabled;
	}

	void setup() {
		mode = setuping;
	}

	void calculate_factors() {
		size_t const size = n.size();

		std::vector<Posit> s;
		std::vector<Posit> c;
		
		s.reserve(size);
		c.reserve(size);

		std::cout << "var: " << var << std::endl;

		for(size_t i=0; i<size; i++) {
			s.push_back(0.5*log10(var[i]));	// calculate log(std)
		}

		std::cout << "s: " << s << std::endl;

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

		std::cout << "c: " << c << std::endl;

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

		std::cout << "scale: " << scale << std::endl;

		return;
	}

	void calculate_factors2() {
		size_t const size = n.size();

		std::vector<Posit> std;
		std.reserve(size);

		std::cout << "n: " << n << std::endl;
		std::cout << "var: " << var << std::endl;

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
		aux /= sum_n;

		scale.back() = aux;
		for(Posit& p : acc_scale)
			p = aux;

		std::cout << "scale: " << scale << std::endl;
		std::cout << "acc_scale: " << acc_scale << std::endl;

		return;
	}


	std::vector<size_t> n;
	std::vector<Posit> var;
	std::vector<Posit> scale;
	std::vector<Posit> acc_scale;

private:
	Mode mode = disabled;
};

template <typename Model, typename Loss, typename Posit>
void setup_back_scale(Model& model, Loss& loss, BackScale<Posit>& bs) {
	bs.setup();

	loss.backward(model);	
	bs.calculate_factors();
	//bs.calculate_factors2();

	bs.enable();

	return;
}

#endif /* BACKSCALE_HPP */

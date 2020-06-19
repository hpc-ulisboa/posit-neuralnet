#ifndef SGD_HPP
#define SGD_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../layer/Parameter.hpp"
#include "../tensor/matrix.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename T>
struct SGDOptions {
	SGDOptions() { }

	template<typename optT=float>
	SGDOptions(optT _learning_rate, optT _momentum=0, optT _dampening=0, optT _weight_decay=0, bool _nesterov=false) :
		learning_rate(T(_learning_rate)),
		momentum(T(_momentum)),
		dampening(T(_dampening)),
		weight_decay(T(_weight_decay)),
		nesterov(_nesterov)
	{ }
	// TODO: protect agains invalid options

	T learning_rate;
	T momentum;
	T dampening;
	T weight_decay;
	bool nesterov;
	bool first = true;
};

template <typename T>
class SGD {
public:
	SGD() { }

	SGD(std::vector<Parameter<T>> parameters0, SGDOptions<T> options0) :
		_parameters(parameters0),
		_options(options0)
	{
		if(_options.momentum!=0){
			_velocities.reserve(_parameters.size());
		}
	}

	void zero_grad() {
		for(Parameter<T>& p : _parameters){
			p.gradient.clear();
		}

		return;
	}

	void step() {
		StdTensor<T> dweight;
		T const pOne(1);

		size_t i=0;
		for(Parameter<T>& p : _parameters){
			//std::cout << "weight = " << p.weight << std::endl;
			
			/*
			// SCALING
			if(LOSS_SCALE!=1)
				p.gradient /= LOSS_SCALE;
			*/

			dweight = p.gradient;

			if(_options.weight_decay != 0){
				//dweight += p.weight * _options.weight_decay;
				fused(dweight, p.weight, pOne, _options.weight_decay);
			}

			if(_options.momentum != 0){
				if(_options.first){
					_velocities.push_back(dweight);
				}
				else{
					//_velocities[i] *= _options.momentum;
					if(_options.dampening!=0) {
						//_velocities[i] += dweight * (1 - _options.dampening);
						fused(_velocities[i], dweight, _options.momentum, pOne-_options.dampening);
					}
					else{
						//_velocities[i] += dweight;
						fused(_velocities[i], dweight, _options.momentum, pOne);
					}
				}
				if(_options.nesterov){
					//dweight += _velocities[i] * _options.momentum;
					fused(dweight, _velocities[i], pOne, _options.momentum);
				}
				else{
					dweight = _velocities[i];
				}
				i++;
			}

			//dweight *= _options.learning_rate;
			//p.weight -= dweight;
			fused(p.weight, dweight, pOne, -_options.learning_rate);
		}

		_options.first = false;

		return;
	}

	SGDOptions<T> options() {
		return _options;
	}

private:
	std::vector<Parameter<T>> _parameters;
	SGDOptions<T> _options;	
	std::vector<StdTensor<T>> _velocities;
};

#endif /* SGD_HPP */

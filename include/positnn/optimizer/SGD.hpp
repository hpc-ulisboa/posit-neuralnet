#ifndef SGD_HPP
#define SGD_HPP

// Custom headers
#include "../layer/Parameter.hpp"
#include "../optimizer/Optimizer.hpp"
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
};

template <typename T>
class SGD : public Optimizer<T>{
public:
	SGD() { }

	SGD(std::vector<Parameter<T>> parameters0, SGDOptions<T> options0) :
		Optimizer<T>(parameters0),
		_options(options0)
	{
		if(_options.momentum!=0){
			_velocities.resize(parameters0.size());
		}
	}

	SGDOptions<T>& options() {
		return _options;
	}

private:

	void update_parameter(Parameter<T>& p, size_t const i) override {
		T const pOne(1);
		StdTensor<T> dweight = p.gradient;

		if(_options.weight_decay != 0){
			//dweight += p.weight * _options.weight_decay;
			//fused(dweight, p.weight, pOne, _options.weight_decay);
			fused(p.weight, dweight, dweight, _options.weight_decay);
		}

		if(_options.momentum != 0){
			if(_velocities[i].empty()){
				_velocities[i] = dweight;
			}
			else{
				//_velocities[i] *= _options.momentum;
				if(_options.dampening!=0) {
					//_velocities[i] += dweight * (1 - _options.dampening);
					fused(_velocities[i], dweight, _options.momentum, pOne-_options.dampening);
				}
				else{
					//_velocities[i] += dweight;
					//fused(_velocities[i], dweight, _options.momentum, pOne);
					fused(_velocities[i], dweight, _velocities[i], _options.momentum);
				}
			}
			if(_options.nesterov){
				//dweight += _velocities[i] * _options.momentum;
				//fused(dweight, _velocities[i], pOne, _options.momentum);
				fused(_velocities[i], dweight, dweight, _options.momentum);
			}
			else{
				dweight = _velocities[i];
			}
		}

		//dweight *= _options.learning_rate;
		//p.weight -= dweight;
		//fused(p.weight, dweight, pOne, -_options.learning_rate);
		fused(dweight, p.weight, p.weight, -_options.learning_rate);

		// When using mixed precision, update different weights
		p.update();

		return;
	}

	SGDOptions<T> _options;	
	std::vector<StdTensor<T>> _velocities;
};

#endif /* SGD_HPP */

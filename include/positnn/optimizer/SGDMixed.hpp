#ifndef SGDMIXED_HPP
#define SGDMIXED_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "../layer/Parameter.hpp"
#include "../optimizer/SGD.hpp"
#include "../tensor/matrix.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <class T1, class T2>
class SGDMixed {
public:
	SGDMixed() { }

	SGDMixed(	std::vector<Parameter<T1>> parameters_model0,
				std::vector<Parameter<T2>> parameters_optimizer0,
				SGDOptions<T2> options0	) :
		_parameters_model(parameters_model0),
		_parameters_optimizer(parameters_optimizer0),
		_options(options0)
	{
		if(_options.momentum!=0){
			_velocities.reserve(_parameters_optimizer.size());
		}
	}

	void zero_grad() {
		for(Parameter<T1>& p : _parameters_model){
			p.gradient.clear();
		}

		return;
	}

	void step() {
		copy_gradients(_parameters_model, _parameters_optimizer);

		StdTensor<T2> dweight;
		T2 const pOne(1);

		size_t i=0;
		for(Parameter<T2>& p : _parameters_optimizer){
			//std::cout << "weight = " << p.weight << std::endl;
			
			dweight = p.gradient;

			if(_options.weight_decay != 0){
				//dweight += p.weight * _options.weight_decay;
				////fused(dweight, p.weight, pOne, _options.weight_decay);
				fused(p.weight, dweight, dweight, _options.weight_decay);
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
						////fused(_velocities[i], dweight, _options.momentum, pOne);
						fused(_velocities[i], dweight, _velocities[i], _options.momentum);
					}
				}
				if(_options.nesterov){
					//dweight += _velocities[i] * _options.momentum;
					////fused(dweight, _velocities[i], pOne, _options.momentum);
					fused(_velocities[i], dweight, dweight, _options.momentum);
				}
				else{
					dweight = _velocities[i];
				}
				i++;
			}

			//dweight *= _options.learning_rate;
			//p.weight -= dweight;
			////fused(p.weight, dweight, pOne, -_options.learning_rate);
			fused(dweight, p.weight, p.weight, -_options.learning_rate);
		}

		_options.first = false;

		copy_parameters(_parameters_optimizer, _parameters_model);

		return;
	}

	SGDOptions<T2> options() {
		return _options;
	}

private:
	std::vector<Parameter<T1>> _parameters_model;
	std::vector<Parameter<T2>> _parameters_optimizer;
	SGDOptions<T2> _options;	
	std::vector<StdTensor<T2>> _velocities;
};

#endif /* SGDMIXED_HPP */

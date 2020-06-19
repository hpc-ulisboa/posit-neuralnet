#ifndef LAYER_HPP
#define LAYER_HPP

// General headers
#include <cmath>

// Custom headers
#include "init.hpp"
#include "Parameter.hpp"
#include "../tensor/StdTensor.hpp"

template <typename Posit>
class Layer {
public:
	Layer()	{}
	virtual ~Layer() {}

	//virtual StdTensor<Posit> forward(StdTensor<Posit> x) {}

	//virtual StdTensor<Posit> backward(StdTensor<Posit> deltaN) {}

	void zero_grad() {
		for(Parameter<Posit>& p : _parameters){
			p.gradient.clear();
		}
	}

	std::vector<Parameter<Posit>>& parameters() {
		return _parameters;
	}

	void register_parameter(StdTensor<Posit>& _weight, StdTensor<Posit>& _gradient) {
		Parameter<Posit> p(_weight, _gradient);
		_parameters.push_back(p);
	}

	void register_parameter(Parameter<Posit>& p) {
		_parameters.push_back(p);
	}

	void register_parameter(std::vector<Parameter<Posit>>& ps) {
		for(Parameter<Posit>& p : ps){
			_parameters.push_back(p);
		}
	}

	void register_module(Layer<Posit>& layer) {
		register_parameter(layer.parameters());
		modules.push_back(&layer);	
	}

	template <typename PositFile=Posit>
	void write(std::ostream& out) {
		for(Parameter<Posit>& p : _parameters)
			p.weight.template write<PositFile>(out);
	}

	template <typename PositFile=Posit>
	void read(std::istream& in) {
		for(Parameter<Posit>& p : _parameters)
			p.weight.template read<PositFile>(in);
	}

	void train() {
		training = true;
		for(Layer<Posit>* layer : modules)
			layer->train();
	}

	void eval() {
		training = false;
		for(Layer<Posit>* layer : modules)
			layer->eval();
	}

protected:
	std::vector<Parameter<Posit>> _parameters;
	std::vector<Layer<Posit>*> modules;
	bool training = false;
};

#endif /* LAYER_HPP */

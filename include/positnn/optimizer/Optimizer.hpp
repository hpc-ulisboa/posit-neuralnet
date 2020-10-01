#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

// General headers
#ifdef USING_LL_THREADS
#include <thread>
#endif /* USING_LL_THREADS */

// Custom headers
#include "../layer/Parameter.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename T>
class Optimizer {
public:
	Optimizer() { }
	virtual ~Optimizer() { }

	Optimizer(std::vector<Parameter<T>> parameters0) :
		_parameters(parameters0)
	{ }

	void zero_grad() {
		for(Parameter<T>& p : _parameters){
			p.gradient.clear();
		}

		return;
	}

#ifndef USING_LL_THREADS

	void step() {
		for(size_t i=0, size=_parameters.size(); i<size; i++) {
			update_parameter(_parameters[i], i);
		}

		return;
	}

#else
	
	void step() {
		const size_t size = _parameters.size();

		// Declare threads (each thread will take care of the same # of parameters)
		const size_t max_threads = (LL_THREADS<size) ? LL_THREADS : size;
		std::vector<std::thread> threads;
		threads.reserve(max_threads);

		// Calculate load for each thread
		size_t const n_samples = size / max_threads;
		size_t const nthreads_more = size % max_threads;

		size_t begin = 0;

		for(size_t t=0; t<max_threads; t++){
			// Get number of samples for this thread
			size_t const thread_samples = (t<nthreads_more) ? n_samples+1 : n_samples;

			threads.push_back(std::thread(&Optimizer::step_thread, this,
										begin, thread_samples	));
			
			// Go to next samples
			begin += thread_samples;
		}

		for(std::thread& t : threads) {
			t.join();
		}	

		return;
	}

protected:

	void step_thread(size_t begin, size_t n) {
		for(size_t i=begin, end=begin+n; i<end; i++) {
			update_parameter(_parameters[i], i);
		}
	}

#endif /* USING_LL_THREADS */

	virtual void update_parameter(Parameter<T>&, size_t const) { }

	std::vector<Parameter<T>> _parameters;
};

#endif /* OPTIMIZER_HPP */

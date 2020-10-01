#ifndef DROPOUT_HPP
#define DROPOUT_HPP

// General headers
#include <random>
#include <universal/posit/posit>

// Custom headers
#include "Layer.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename OptimizerT>
class Dropout : public Layer<OptimizerT> {
public:
	Dropout(float _p=0.5) :
		p(_p),
		distribution(_p)
	{
		generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	}

	template <typename T>
	StdTensor<T> forward(StdTensor<T>& x) {
		if(Layer<OptimizerT>::training) {
			zero.resize(x.size());
			std::generate(zero.begin(), zero.end(), [&]{ return distribution(generator); });

			return dropout(x);
		}

		return x;
	}

	template <typename T>
	StdTensor<T> backward(StdTensor<T>& x) {
		if(x.size() != zero.size()) {
			std::cerr << "ERROR: size of x should be " << zero.size() << " instead of " << x.size() << std::endl;
			return x;
		}

		return dropout(x);
	}

private:

#ifndef USING_LL_THREADS

	template <typename T>
	StdTensor<T> dropout(StdTensor<T> x) {
		T const scale = 1/(1-T(p));

		for(size_t i=0, size=x.size(); i<size; i++) {
			if(zero[i]){
				x[i].setzero();
			}
			else{
				x[i] *= scale;
			}
		}

		return x;
	}

#else

	template <typename T>
	StdTensor<T> dropout(StdTensor<T> x) {
		T const scale = 1/(1-T(p));

		const size_t size = x.size();

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

			threads.push_back(std::thread(&Dropout::dropout_thread<T>, this,
										std::ref(x), std::cref(scale),
										begin, thread_samples	));
			
			// Go to next samples
			begin += thread_samples;
		}

		for(std::thread& t : threads) {
			t.join();
		}	

		return x;
	}

	template <typename T>
	void dropout_thread(StdTensor<T>& x, T const& scale, size_t begin, size_t n) {
		for(size_t i=begin, end=begin+n; i<end; i++) {
			if(zero[i]){
				x[i].setzero();
			}
			else{
				x[i] *= scale;
			}
		}

		return;
	}

#endif /* USING_LL_THREADS */

	float p;
	std::default_random_engine generator;
	std::bernoulli_distribution distribution;
	std::vector<bool> zero;
};

#endif /* DROPOUT_HPP */

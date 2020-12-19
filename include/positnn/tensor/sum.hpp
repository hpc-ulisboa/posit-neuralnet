#ifndef SUM_HPP
#define SUM_HPP

#ifdef LL_THREADS
	#if LL_THREADS>1
		#define USING_LL_THREADS
	#endif
#endif /* LL_THREADS */

// General headers
#ifdef USING_LL_THREADS
#include <thread>
#endif /* USING_LL_THREADS */
#include <universal/posit/posit>
#include <vector>

// Custom headers
#include "StdTensor.hpp"
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

#ifdef USING_LL_THREADS

// Function to be executed by each thread to sum first axis
template <size_t nbits, size_t es>
void sum_first_thread(	StdTensor<posit<nbits, es>> const& input,
						StdTensor<posit<nbits, es>>& output,
						size_t const i_begin, size_t const nelem	){

	size_t const i_end = i_begin+nelem;
	size_t const size = input.size();
	size_t const stride = input.strides()[0];

	Quire<nbits, es> q;

	// Loop through output elements
	for(size_t i=i_begin; i<i_end; i++) {
		q.clear();
		
		// Loop thorugh first axis of input
		for(size_t j=i; j<size; j+=stride){
			q += input[j];
		}

		convert(q.to_value(), output[i]);
	}

	return;
}

// Matrix sum along first axis using threads
template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> sum_first (const StdTensor<posit<nbits, es>>& input){
	std::vector<size_t> new_shape;
	
	if(input.dim()==1)
		new_shape.push_back(1);
	else
		new_shape = std::vector<size_t>(input.shape().begin()+1, input.shape().end());

	// TODO: THROW ERROR IF SUM DIMENSIONS ARE INVALID
	StdTensor<posit<nbits, es>> output(new_shape);
	const size_t size = output.size();

	const size_t max_threads = (LL_THREADS<size) ? LL_THREADS : size;
	std::vector<std::thread> threads;
	threads.reserve(max_threads);

	size_t const nelem0 = size / max_threads;
	size_t const nthreads_more = size % max_threads;

	size_t i_begin = 0;

	for(size_t t=0; t<max_threads; t++){
		size_t nelem = nelem0;
		if(t < nthreads_more)
			nelem++;

		threads.push_back(std::thread(sum_first_thread<nbits, es>,
										std::cref(input), std::ref(output),
										i_begin, nelem));

		i_begin += nelem;
	}

	for(std::thread& t : threads) {
		t.join();
	}

	return output;
}

#else

// Matrix sum along first axis
template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> sum_first(const StdTensor<posit<nbits, es>>& input){
	std::vector<size_t> new_shape;
	
	if(input.dim()==1)
		new_shape.push_back(1);
	else
		new_shape = std::vector<size_t>(input.shape().begin()+1, input.shape().end());

	StdTensor<posit<nbits, es>> output(new_shape);

	size_t const size = input.size();
	size_t const stride = input.strides()[0];

	Quire<nbits, es> q;	// TODO: COMPUTE BEST CAPACITY

	// Loop through output elements
	for(size_t i=0; i<stride; i++) {
		q.clear();
		
		// Loop thorugh first axis of input
		for(size_t j=i; j<size; j+=stride){
			q += input[j];
		}

		convert(q.to_value(), output[i]);
	}

	return output;
}

#endif /* USING_LL_THREADS */

#ifdef USING_LL_THREADS

// Function to be executed by each thread to sum last two axes
template <size_t nbits, size_t es>
void sum_last2_thread(	StdTensor<posit<nbits, es>> const& input,
						StdTensor<posit<nbits, es>>& output,
						size_t const input_begin, size_t const output_begin,
						size_t const nelem	){

	size_t const output_end = output_begin+nelem;
	size_t const size = output.size();
	size_t const stride = (input.dim()>2) ? input.strides()[input.dim()-3] : size;

	Quire<nbits, es> q;
	size_t begin = input_begin;
	size_t end = input_begin+stride;
	
	// Loop through output elements
	for(size_t n=output_begin; n<output_end; n++) {
		q.clear();
		
		// Loop through axes to sum
		for(size_t j=begin; j<end; j++){
			q += input[j];
		}

		convert(q.to_value(), output[n]);

		begin = end;
		end += stride;
	}

	return;
}

// Sum of last two axes using threads
template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> sum_last2 (const StdTensor<posit<nbits, es>>& input){
	// TODO: THROW ERROR IF SUM DIMENSIONS ARE INVALID
	
	std::vector<size_t> new_shape;
	
	if(input.dim()==2)
		new_shape.push_back(1);
	else
		new_shape = std::vector<size_t>(input.shape().begin(), input.shape().end()-2);

	StdTensor<posit<nbits, es>> output(new_shape);
	size_t const size = output.size();
	size_t const stride = (input.dim()>2) ? input.strides()[input.dim()-3] : size;

	const size_t max_threads = (LL_THREADS<size) ? LL_THREADS : size;
	std::vector<std::thread> threads;
	threads.reserve(max_threads);

	size_t const nelem0 = size / max_threads;
	size_t const nthreads_more = size % max_threads;
	size_t const stride0 = nelem0 * stride;

	size_t input_begin = 0;
	size_t output_begin = 0;

	for(size_t t=0; t<max_threads; t++){
		size_t nelem = nelem0;
		if(t < nthreads_more)
			nelem++;

		threads.push_back(std::thread(sum_last2_thread<nbits, es>,
										std::cref(input), std::ref(output),
										input_begin, output_begin, nelem));

		input_begin += stride0;
		output_begin += nelem;

		if(t < nthreads_more){
			input_begin += stride;
		}
	}

	for(std::thread& t : threads) {
		t.join();
	}

	return output;
}

#else

// Sum of last two axes
template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> sum_last2(StdTensor<posit<nbits, es>> const& input) {
	// TODO: error if dim < 2

	std::vector<size_t> new_shape;

	if(input.dim()==2)
		new_shape.push_back(1);
	else
		new_shape = std::vector<size_t>(input.shape().begin(), input.shape().end()-2);

	StdTensor<posit<nbits, es>> output(new_shape);

	size_t const size = output.size();
	size_t const stride = (input.dim()>2) ? input.strides()[input.dim()-3] : size;

	Quire<nbits, es> q;	// TODO: COMPUTE BEST CAPACITY
	size_t begin = 0;
	size_t end = stride;

	// Loop through output
	for(size_t n=0; n<size; n++) {
		q.clear();
		
		// Loop through axes to sum
		for(size_t j=begin; j<end; j++){
			q += input[j];
		}

		convert(q.to_value(), output[n]);

		begin = end;
		end += stride;
	}

	return output;
}

#endif /* USING_LL_THREADS */

/*

// Matrix sum along axis
template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> sum(const StdTensor<posit<nbits, es>>& a, const size_t axis=0){
// ERROR: LOOP J IS NOT CORRECT
	// constexpr size_t nbits = T::nbits;
	// constexpr size_t es = T::es;
	
	std::vector<size_t> new_shape = a.shape();
	new_shape.erase(new_shape.begin()+axis);
	if (new_shape.empty())
		new_shape.push_back(1);

	StdTensor<posit<nbits, es>> c(new_shape);

	Quire<nbits, es> q;	// TODO: COMPUTE BEST CAPACITY
	
	size_t n=0;
	size_t const size = a.size();
	size_t const axis_size = a.shape()[axis];
	size_t const stride = a.strides()[axis];
	size_t const loop_stride = axis_size*stride;

	for(size_t i=0; i<size; i+=loop_stride) {	// loop blocks
		for(size_t j=0; j<stride; j++){	// loop beginning elements of block
			q.clear();
			for(size_t k=i+j, l=0; l<axis_size; k+=stride, l++){	// loop elements to sum
				q += a[k];
			}
			convert(q.to_value(), c[n++]);
		}
	}

	return c;
}

*/

#endif /* SUM */

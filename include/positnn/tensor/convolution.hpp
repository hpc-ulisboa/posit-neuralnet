#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

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
#include "Window.hpp"

// Namespaces
using namespace sw::unum;

// Algorithm for a convolution
template <size_t nbits, size_t es, size_t capacity=nbits-1>
void do_convolution(StdTensor<posit<nbits, es>> const& input,
					StdTensor<posit<nbits, es>> const& kernel,
					quire<nbits, es, capacity>& output, Window const& w,
					size_t const input_idx, size_t const kernel_idx, size_t const idx){

	// Begin and end element to operate (multiply)
	size_t const begin = w.window_idx[idx];
	size_t const end = w.window_idx[idx+1];

	// If there is no overlap between input and kernel
	if(begin == end)
		return;

	// Loop through elements to operate with input and kernel
	for(size_t i=begin; i<end; i++){
		size_t input_i = input_idx + w.map_window[i];
		size_t kernel_i = kernel_idx + w.kernel_window[i];

		output += quire_mul(input[input_i], kernel[kernel_i]); 
	}

	return;
}

#ifdef USING_LL_THREADS

template <size_t nbits, size_t es, size_t capacity=nbits-1>
void convolution2d_thread(	StdTensor<posit<nbits, es>> const& input,
							StdTensor<posit<nbits, es>> const& weight,
							StdTensor<posit<nbits, es>> const& bias,
							StdTensor<posit<nbits, es>>& output,
							Window const* w,
							size_t input_batch, size_t output_batch,
							size_t const n_samples	){
	
	// Check if bias is empty
	bool const no_bias = bias.empty();

	// Get number of input and output channels
	size_t const output_channels = weight.shape()[0];
	size_t const input_channels = weight.shape()[1];

	// Strides to loop tensors
	size_t const input_batch_stride = input.strides()[0];
	size_t const input_channel_stride = input.strides()[1];
	size_t const output_batch_stride = output.strides()[0];
	size_t const output_channel_stride = output.strides()[1];
	size_t const weight_out_channel_stride = weight.strides()[0];
	size_t const weight_in_channel_stride = weight.strides()[1];

	// Size of output matrix after convolution
	size_t const size = output_channel_stride;

	// Initialize quire
	quire<nbits, es, capacity> q;

	// Loop through batch
	for(size_t i=0; i<n_samples; i++){
		size_t weight_out_channel = 0;
		size_t output_channel = output_batch;

		// Loop through output channels
		for(size_t j=0; j<output_channels; j++){

			// Loop through output rows and cols
			for(size_t idx=0; idx<size; idx++){

				// Indices of input and weight for input channel
				size_t input_channel = input_batch;
				size_t weight_in_channel = weight_out_channel;
	
				// Set quire to bias value
				if(no_bias)
					q.reset();
				else
					q = bias[j];

				// Loop through input channels
				for(size_t channel=0; channel<input_channels; channel++){
					// Compute convolution for that block
					do_convolution(	input, weight, q, *w,
									input_channel, weight_in_channel, idx	);

					// Loop through input channels
					input_channel += input_channel_stride;
					weight_in_channel += weight_in_channel_stride;
				}
				
				// Convert result from quire to posit
				convert(q.to_value(), output[output_channel+idx]);
			}
				
			weight_out_channel += weight_out_channel_stride;
			output_channel += output_channel_stride;
		}

		input_batch += input_batch_stride;
		output_batch += output_batch_stride;
	}
}

template <size_t nbits, size_t es, size_t capacity=nbits-1>
StdTensor<posit<nbits, es>> convolution2d(	StdTensor<posit<nbits, es>> const& input,
											StdTensor<posit<nbits, es>> const& weight,
											StdTensor<posit<nbits, es>> const& bias,
											size_t const stride=1,
											size_t const padding=0,
											Window* w=NULL ){
	
	// TODO: throw error if kernel and input have different in_channels
	// TODO: check other errors
	
	// Get windows
	bool empty = (w==NULL);

	if(empty)
		w = new Window();

	if(!w->initialized){
		w->forward(input.shape()[2], input.shape()[3],
					weight.shape()[2], weight.shape()[3],
					stride, padding	);
	}

	// Get batch size and # of output channels
	size_t const batch_size = input.shape()[0];
	size_t const output_channels = weight.shape()[0];

	// Create tensor for output
	StdTensor<posit<nbits, es>> output({batch_size, output_channels, w->output_height, w->output_width});
	
	// Strides to loop tensors
	size_t const input_batch_stride = input.strides()[0];
	size_t const output_batch_stride = output.strides()[0];

	// Distribute threads (each thread will take care of the same # of samples)
	const size_t max_threads = (LL_THREADS<batch_size) ? LL_THREADS : batch_size;
	std::vector<std::thread> threads;
	threads.reserve(max_threads);

	// Calculate load for each thread
	size_t const n_samples = batch_size / max_threads;
	size_t const nthreads_more = batch_size % max_threads;

	// Strides to jump between threads' samples
	size_t const input_samples_stride = n_samples * input_batch_stride; 
	size_t const output_samples_stride = n_samples * output_batch_stride; 

	// Start at first sample
	size_t input_samples_begin = 0;
	size_t output_samples_begin = 0;

	for(size_t t=0; t<max_threads; t++){

		// Get number of samples for this thread
		size_t thread_samples = n_samples;
		if(t < nthreads_more)
			thread_samples++;

		threads.push_back(std::thread(convolution2d_thread<nbits, es, capacity>,
										std::cref(input), std::cref(weight), std::cref(bias), std::ref(output), w,
										input_samples_begin, output_samples_begin, thread_samples	));
		
		// Go to next samples
		input_samples_begin += input_samples_stride;
		output_samples_begin += output_samples_stride;

		if(t < nthreads_more){
			input_samples_begin += input_batch_stride;
			output_samples_begin += output_batch_stride;
		}
	}
	
	for(std::thread& t : threads) {
		t.join();
	}	

	if(empty)
		delete w;

	return output;

}

#else

template <size_t nbits, size_t es, size_t capacity=nbits-1>
StdTensor<posit<nbits, es>> convolution2d(	StdTensor<posit<nbits, es>> const& input,
											StdTensor<posit<nbits, es>> const& weight,
											StdTensor<posit<nbits, es>> const& bias,
											size_t const stride=1,
											size_t const padding=0,
											Window* w=NULL ){
	
	// TODO: throw error if kernel and input have different in_channels
	// TODO: check other errors
	
	bool empty = (w==NULL);

	if(empty)
		w = new Window();

	if(!w->initialized){
		w->forward(input.shape()[2], input.shape()[3],
					weight.shape()[2], weight.shape()[3],
					stride, padding	);
	}
	
	// Check if bias is empty
	bool const no_bias = bias.empty();

	size_t const batch_size = input.shape()[0];
	size_t const output_channels = weight.shape()[0];
	size_t const input_channels = weight.shape()[1];

	StdTensor<posit<nbits, es>> output({batch_size, output_channels, w->output_height, w->output_width});

	size_t const input_batch_stride = input.strides()[0];
	size_t const input_channel_stride = input.strides()[1];
	size_t const weight_out_channel_stride = weight.strides()[0];
	size_t const weight_in_channel_stride = weight.strides()[1];
	size_t const output_batch_stride = output.strides()[0];
	size_t const output_channel_stride = output.strides()[1];

	size_t const size = output_channel_stride;

	size_t input_batch = 0;
	size_t output_batch = 0;

	// Initialize quire
	quire<nbits, es, capacity> q;

	// Loop through batch
	for(size_t i=0; i<batch_size; i++){
		size_t weight_out_channel = 0;
		size_t output_channel = output_batch;

		// Loop through output channels
		for(size_t j=0; j<output_channels; j++){

			// Loop through output rows and cols
			for(size_t idx=0; idx<size; idx++){

				// Indices of input and weight for input channel
				size_t input_channel = input_batch;
				size_t weight_in_channel = weight_out_channel;
	
				// Set quire to bias value
				if(no_bias)
					q.reset();
				else
					q = bias[j];

				// Loop through input channels
				for(size_t channel=0; channel<input_channels; channel++){
					// Compute convolution for that block
					do_convolution(	input, weight, q, *w,
									input_channel, weight_in_channel, idx	);

					input_channel += input_channel_stride;
					weight_in_channel += weight_in_channel_stride;
				}
				
				// Convert result from quire to posit
				convert(q.to_value(), output[output_channel+idx]);
			}
				
			weight_out_channel += weight_out_channel_stride;
			output_channel += output_channel_stride;
		}

		input_batch += input_batch_stride;
		output_batch += output_batch_stride;
	}

	if(empty)
		delete w;

	return output;
}

#endif /* USING_LL_THREADS */

#ifdef USING_LL_THREADS

template <size_t nbits, size_t es, size_t capacity=nbits-1>
void convolution2d_gradient_thread(	StdTensor<posit<nbits, es>> const& input,
									StdTensor<posit<nbits, es>> const& delta,
									StdTensor<posit<nbits, es>>& dweight,
									Window const* w,
									size_t const dweight_begin, size_t const n_samples	){

	//std::cout << "dweight_begin: " << dweight_begin << std::endl;

	size_t const batch_size = input.shape()[0];
	size_t const input_channels = input.shape()[1];
	size_t const output_channels = delta.shape()[1];
	
	// Strides to loop tensors
	size_t const input_batch_stride = input.strides()[0];
	size_t const input_channel_stride = input.strides()[1];
	size_t const delta_batch_stride = delta.strides()[0];
	size_t const delta_channel_stride = delta.strides()[1];

	size_t const size = dweight.strides()[1];

	//std::cout << "strides: " << dweight.strides() << std::endl;
	
	// Initial indices
	size_t out_channel0 = dweight_begin / dweight.strides()[0];
	size_t in_channel0 = dweight_begin % dweight.strides()[0];
	in_channel0 /= dweight.strides()[1];
	size_t idx0 = dweight_begin % dweight.strides()[1];

	/*
	std::cout << "out_channel0: " << out_channel0 << std::endl;
	std::cout << "in_channel0: " << in_channel0 << std::endl;
	std::cout << "idx0: " << idx0 << std::endl;
	*/
	
	// Initialize quire
	quire<nbits, es, capacity> q;

	// First indices
	size_t input_channel0 = in_channel0 * input_channel_stride;
	size_t delta_channel = out_channel0 * delta_channel_stride;
	size_t n = dweight_begin;
	
	// Final index
	size_t const n_end = dweight_begin + n_samples;

	bool first = true;

	// Loop through output (delta) channels
	for(size_t i=out_channel0; i<output_channels; i++){

		// Index of input channel
		size_t input_channel = (first) ? input_channel0 : 0;
		
		// Loop through input channels
		for(size_t j = (first) ? in_channel0 : 0; j<input_channels; j++){

			// Loop through weights rows and cols
			for(size_t idx = (first) ? idx0 : 0; idx<size; idx++){

				// Indices of samples of delta and input
				size_t input_batch = input_channel;
				size_t delta_batch = delta_channel;
	
				// Set quire to bias value
				q.reset();

				// Loop through batch
				for(size_t batch=0; batch<batch_size; batch++){
					// Compute convolution for that block
					do_convolution(	input, delta, q, *w,
									input_batch, delta_batch, idx	);

					input_batch += input_batch_stride;
					delta_batch += delta_batch_stride;
				}

				//std::cout << "out channel: " << i << "\t in channel: " << j << "\t idx: " << idx << "\t n: " << n << std::endl;

				// Convert result from quire to posit
				convert(q.to_value(), dweight[n++]);
				//getchar();

				if(n == n_end)
					return;

				if(first)
					first = false;
			}
				
			input_channel += input_channel_stride;
		}

		delta_channel += delta_channel_stride;
	}
}

template <size_t nbits, size_t es, size_t capacity=nbits-1>
StdTensor<posit<nbits, es>> convolution2d_gradient(
												StdTensor<posit<nbits, es>> const& input,
												StdTensor<posit<nbits, es>> const& delta,
												size_t const stride=1,
												size_t const padding=0,
												Window* w=NULL ){
	
	// TODO: throw error if kernel and input have different in_channels
	// TODO: check other errors
	
	bool empty = (w==NULL);

	if(empty)
		w = new Window();

	if(!w->initialized){
		w->forward(input.shape()[2], input.shape()[3],
					delta.shape()[2], delta.shape()[3],
					stride, padding	);
	}

	size_t const input_channels = input.shape()[1];
	size_t const output_channels = delta.shape()[1];

	StdTensor<posit<nbits, es>> dweight({output_channels, input_channels, w->output_height, w->output_width});

	size_t const size = dweight.size();

	// Distribute threads (each thread will take care of the same # of samples)
	const size_t max_threads = (LL_THREADS<size) ? LL_THREADS : size;
	std::vector<std::thread> threads;
	threads.reserve(max_threads);

	// Calculate load for each thread
	size_t const n_samples = size / max_threads;
	size_t const nthreads_more = size % max_threads;

	// Start at first sample
	size_t dweight_begin = 0;

	for(size_t t=0; t<max_threads; t++){

		// Get number of samples for this thread
		size_t thread_samples = n_samples;
		if(t < nthreads_more)
			thread_samples++;

		threads.push_back(std::thread(convolution2d_gradient_thread<nbits, es, capacity>,
										std::cref(input), std::cref(delta), std::ref(dweight), w,
										dweight_begin, thread_samples	));
		
		// Go to next dweight element
		dweight_begin += thread_samples;

		threads.back().join();
	}
	
	/*
	for(std::thread& t : threads) {
		t.join();
	}
	*/

	if(empty)
		delete w;

	return dweight;

}

#else

template <size_t nbits, size_t es, size_t capacity=nbits-1>
StdTensor<posit<nbits, es>> convolution2d_gradient(
												StdTensor<posit<nbits, es>> const& input,
												StdTensor<posit<nbits, es>> const& delta,
												size_t const stride=1,
												size_t const padding=0,
												Window* w=NULL ){
	
	// TODO: throw error if kernel and input have different in_channels
	// TODO: check other errors
	
	bool empty = (w==NULL);

	if(empty)
		w = new Window();

	if(!w->initialized){
		w->forward(input.shape()[2], input.shape()[3],
					delta.shape()[2], delta.shape()[3],
					stride, padding	);
	}

	size_t const batch_size = input.shape()[0];
	size_t const input_channels = input.shape()[1];
	size_t const output_channels = delta.shape()[1];

	StdTensor<posit<nbits, es>> dweight({output_channels, input_channels, w->output_height, w->output_width});

	size_t const input_batch_stride = input.strides()[0];
	size_t const input_channel_stride = input.strides()[1];
	size_t const delta_batch_stride = delta.strides()[0];
	size_t const delta_channel_stride = delta.strides()[1];

	size_t const size = dweight.strides()[1];

	size_t delta_channel = 0;
	size_t n=0;

	// Initialize quire
	quire<nbits, es, capacity> q;

	// Loop through output (delta) channels
	for(size_t i=0; i<output_channels; i++){
		
		// Index of input channel
		size_t input_channel = 0;

		// Loop through input channels
		for(size_t j=0; j<input_channels; j++){

			// Loop through weights rows and cols
			for(size_t idx=0; idx<size; idx++){

				// Indices of samples of delta and input
				size_t input_batch = input_channel;
				size_t delta_batch = delta_channel;
	
				// Set quire to bias value
				q.reset();

				// Loop through batch
				for(size_t batch=0; batch<batch_size; batch++){
					// Compute convolution for that block
					do_convolution(	input, delta, q, *w,
									input_batch, delta_batch, idx	);

					input_batch += input_batch_stride;
					delta_batch += delta_batch_stride;

				}
				
				// Convert result from quire to posit
				convert(q.to_value(), dweight[n++]);
			}
				
			input_channel += input_channel_stride;
		}

		delta_channel += delta_channel_stride;
	}

	if(empty)
		delete w;

	return dweight;
}

#endif /* USING_LL_THREADS */

template <typename T>
StdTensor<T> rotate_weight(StdTensor<T> const& input) {
	StdTensor<T> output({	input.shape()[1],
							input.shape()[0],
							input.shape()[2],
							input.shape()[3]	});

	// Strides
	size_t out_channel_stride = output.strides()[0];
	size_t in_channel_stride = output.strides()[1];

	// Sizes
	size_t size = output.size();
	size_t nelem = in_channel_stride;

	// Index of input
	size_t i = 0;

	// Loop through output in channels
	for(size_t in_channel=0; in_channel<out_channel_stride; in_channel+=in_channel_stride){
		// Loop through output out channels
		for(size_t out_channel=in_channel; out_channel<size; out_channel+=out_channel_stride){
			// Loop through elements of channel of output
			for(size_t idx = out_channel+nelem; idx --> out_channel; ){
				output[idx] = input[i++];
			}
		}
	}

	return output;
}

#endif /* CONVOLUTION_HPP */

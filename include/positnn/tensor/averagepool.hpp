#ifndef AVERAGEPOOL_HPP
#define AVERAGEPOOL_HPP

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
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

template <size_t nbits, size_t es>
void do_avgpool2d(	StdTensor<posit<nbits, es>> const& input,
					posit<nbits, es>& output,
					Window const& w, size_t kernel_size,
					size_t const input_idx, size_t const idx	){
	
	// Begin and end element to operate (multiply)
	size_t const begin = w.window_idx[idx];
	size_t const end = w.window_idx[idx+1];

	// If there is no overlap between input and kernel
	if(begin == end)
		return;

	// Element only appears in 1 window
	if(begin+1 == end){
		size_t input_i = input_idx + w.map_window[begin];
		output = input[input_i];
	}
	// More than 1 element
	else{
		// Initialize Quire
		Quire<nbits, es> q;

		// Loop through elements to operate with input and kernel
		for(size_t i=begin; i<end; i++){
			size_t input_i = input_idx + w.map_window[i];

			q += input[input_i]; 
		}

		convert(q.to_value(), output);
	}


	output /= kernel_size;

	return;
}

#ifdef USING_LL_THREADS

template <size_t nbits, size_t es>
void averagepool2d_thread(	StdTensor<posit<nbits, es>> const& input,
							StdTensor<posit<nbits, es>>& output,
							Window const* w, const size_t kernel_total_size,
							size_t input_batch, size_t output_batch,
							size_t const n_samples	){

	// Get number of output channels
	size_t const input_channels = input.shape()[1];

	// Strides to loop tensors
	size_t const input_batch_stride = input.strides()[0];
	size_t const input_channel_stride = input.strides()[1];
	size_t const output_batch_stride = output.strides()[0];
	size_t const output_channel_stride = output.strides()[1];

	size_t const size = output_channel_stride;

	// Loop through batch
	for(size_t i=0; i<n_samples; i++){
		size_t input_channel = input_batch;
		size_t output_channel = output_batch;
		
		// Loop through input channels
		for(size_t j=0; j<input_channels; j++){

			// Loop through output rows and cols
			for(size_t idx=0; idx<size; idx++){

				// Compute average pooling for that block
				do_avgpool2d<nbits, es>(	input, output[output_channel+idx],
													*w,	kernel_total_size,
													input_channel, idx	);
			}
				
			input_channel += input_channel_stride;
			output_channel += output_channel_stride;
		}
				
		input_batch += input_batch_stride;
		output_batch += output_batch_stride;
	}
}

template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> averagepool2d(	StdTensor<posit<nbits, es>> const& input,
											size_t const kernel_size,
											size_t const stride,
											size_t const padding,
											Window* w=NULL	){
	//
	// TODO: throw error if kernel and input have different in_channels
	// TODO: check other errors
	
	// Get windows
	bool empty = (w==NULL);

	if(empty)
		w = new Window();

	if(!w->initialized){
		w->output_to_input(	input.shape()[2], input.shape()[3],
							kernel_size, kernel_size,
							stride, padding	);
	}

	// Get batch size and # of input channels
	size_t const batch_size = input.shape()[0];
	size_t const input_channels = input.shape()[1];

	// Create tensor for output
	StdTensor<posit<nbits, es>> output({batch_size, input_channels, w->output_height, w->output_width});

	// Strides to loop tensors
	size_t const input_batch_stride = input.strides()[0];
	size_t const output_batch_stride = output.strides()[0];

	size_t const kernel_total_size = kernel_size*kernel_size;
	
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

		threads.push_back(std::thread(averagepool2d_thread<nbits, es>,
										std::cref(input), std::ref(output),
										w, kernel_total_size,
										input_samples_begin, output_samples_begin,
										thread_samples	));
		
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

template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> averagepool2d(	StdTensor<posit<nbits, es>> const& input,
											size_t const kernel_size,
											size_t const stride,
											size_t const padding,
											Window* w=NULL	){

	// TODO: throw error if kernel and input have different in_channels
	// TODO: check other errors

	bool empty = (w==NULL);

	if(empty)
		w = new Window();

	if(!w->initialized){
		w->output_to_input(	input.shape()[2], input.shape()[3],
							kernel_size, kernel_size,
							stride, padding	);
	}

	size_t const batch_size = input.shape()[0];
	size_t const input_channels = input.shape()[1];

	StdTensor<posit<nbits, es>> output({batch_size, input_channels, w->output_height, w->output_width});

	size_t const input_batch_stride = input.strides()[0];
	size_t const input_channel_stride = input.strides()[1];
	size_t const output_batch_stride = output.strides()[0];
	size_t const output_channel_stride = output.strides()[1];

	size_t const size = output_channel_stride;

	size_t const kernel_total_size = kernel_size*kernel_size;

	size_t input_batch = 0;
	size_t output_batch = 0;

	// Loop through batch
	for(size_t i=0; i<batch_size; i++){
		size_t input_channel = input_batch;
		size_t output_channel = output_batch;

		// Loop through input channels
		for(size_t j=0; j<input_channels; j++){

			// Loop through output rows and cols
			for(size_t idx=0; idx<size; idx++){

				// Compute average pooling for that block
				do_avgpool2d<nbits, es>(	input, output[output_channel+idx], *w,
													kernel_total_size, input_channel, idx	);
			}
				
			input_channel += input_channel_stride;
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

template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> averagepool2d_backward(	StdTensor<posit<nbits, es>> const& delta,
													std::vector<size_t> const& input_shape,
													size_t const kernel_size,
													size_t const stride,
													size_t const padding,
													Window* w=NULL	){

	// Throw error if stride!=kernel_size
	
	bool empty = (w==NULL);

	if(empty)
		w = new Window();

	if(!w->initialized){
		w->input_to_output(	input_shape[2], input_shape[3],
							kernel_size, kernel_size,
							stride, padding	);
	}

	size_t const kernel_total_size = kernel_size*kernel_size;

	StdTensor<posit<nbits, es>> deltaN_1(input_shape);

	// Strides to loop tensors
	size_t input_channel_stride = deltaN_1.strides()[1];
	size_t output_channel_stride = delta.strides()[1];
	
	// Number of channels (matrices) of input/output
	size_t const total_channels = deltaN_1.shape()[0]*deltaN_1.shape()[1];
	// Number of elements of input matrix (size of channel)
	size_t const size = input_channel_stride;

	size_t input_channel = 0;
	size_t output_channel = 0;

	// Loop through matrices
	for(size_t i=0; i<total_channels; i++) {
		// Loop through elements of matrix (deltaN_1)
		for(size_t idx=0; idx<size; idx++) {

			// Compute average pooling for that block
			do_avgpool2d<nbits, es>(	delta, deltaN_1[input_channel+idx], *w,
												kernel_total_size, output_channel, idx	);
		}

		input_channel += input_channel_stride;
		output_channel += output_channel_stride;
	}

	if(empty)
		delete w;

	return deltaN_1;
}

#endif /* AVERAGEPOOL_HPP */

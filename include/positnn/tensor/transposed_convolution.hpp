#ifndef TRANSPOSED_CONVOLUTION_HPP
#define TRANSPOSED_CONVOLUTION_HPP

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
#include "convolution.hpp"		// reuse do_convolution
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

// ============================================================================
//  A transposed convolution is the same correlation-style loop as a regular
//  convolution with the roles of input and output swapped. Instead of redoing
//  the modulo arithmetic for every output value, the index mapping is unrolled
//  once per geometry into a Window (invalid and zero contributions are dropped
//  at build time) and then traversed flatly by do_convolution, exactly as in
//  convolution.hpp.
//
//  Three windows, all using the same (map_window, kernel_window, window_idx)
//  layout, only the meaning of a bucket differs:
//    forward      bucket = output pixel     -> pairs (input pixel, kernel pos)
//    grad_input   bucket = input pixel      -> pairs (delta pixel, kernel pos)
//    grad_weight  bucket = kernel position  -> pairs (input pixel, delta pixel)
//
//  The windows only depend on the geometry, so the layer caches them as members
//  (like Conv2d w1/w2/w3) and they are built once per training run. All spatial
//  parameters are square, consistent with the rest of the framework.
// ============================================================================

// Output size of a transposed convolution along one axis.
inline size_t transposed_output_size(size_t const in, size_t const kernel,
                                     size_t const stride, size_t const padding,
                                     size_t const output_padding, size_t const dilation) {
	return (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1 + output_padding;
}

// ----------------------------------------------------------------------------
//  Window builders
// ----------------------------------------------------------------------------

// FORWARD: per output pixel, the contributing (input pixel, kernel position).
// out[oy,ox] += sum input[iy,ix]*weight[ky,kx]  with  iy = (oy+pad-ky*dil)/stride
inline void build_transposed_forward_window(Window& w,
                                            size_t const in_h, size_t const in_w,
                                            size_t const out_h, size_t const out_w,
                                            size_t const kernel, size_t const stride,
                                            size_t const padding, size_t const dilation) {
	w.output_height = out_h;
	w.output_width  = out_w;
	w.map_window.clear();
	w.kernel_window.clear();
	w.window_idx.clear();
	w.window_idx.reserve(out_h * out_w + 1);

	ptrdiff_t const s = static_cast<ptrdiff_t>(stride);
	size_t size = 0;

	for(size_t oy=0; oy<out_h; oy++){
		for(size_t ox=0; ox<out_w; ox++){
			w.window_idx.push_back(size);

			for(size_t ky=0; ky<kernel; ky++){
				ptrdiff_t const y_un = static_cast<ptrdiff_t>(oy + padding) - static_cast<ptrdiff_t>(ky * dilation);
				if(y_un < 0 || y_un % s != 0) continue;
				size_t const iy = static_cast<size_t>(y_un / s);
				if(iy >= in_h) continue;

				for(size_t kx=0; kx<kernel; kx++){
					ptrdiff_t const x_un = static_cast<ptrdiff_t>(ox + padding) - static_cast<ptrdiff_t>(kx * dilation);
					if(x_un < 0 || x_un % s != 0) continue;
					size_t const ix = static_cast<size_t>(x_un / s);
					if(ix >= in_w) continue;

					w.map_window.push_back(iy * in_w + ix);		// input pixel, within channel
					w.kernel_window.push_back(ky * kernel + kx);	// kernel position, within ic/oc block
					size++;
				}
			}
		}
	}

	w.window_idx.push_back(size);
	w.map_window.shrink_to_fit();
	w.kernel_window.shrink_to_fit();
	w.window_idx.shrink_to_fit();
	w.initialized = true;
}

// GRAD-INPUT: per input pixel, the contributing (delta pixel, kernel position).
// grad_in[ih,iw] += sum delta[oh,ow]*weight[kh,kw]  with  oh = ih*stride+kh*dil-pad
inline void build_transposed_grad_input_window(Window& w,
                                               size_t const in_h, size_t const in_w,
                                               size_t const out_h, size_t const out_w,
                                               size_t const kernel, size_t const stride,
                                               size_t const padding, size_t const dilation) {
	w.output_height = in_h;
	w.output_width  = in_w;
	w.map_window.clear();
	w.kernel_window.clear();
	w.window_idx.clear();
	w.window_idx.reserve(in_h * in_w + 1);

	size_t size = 0;

	for(size_t ih=0; ih<in_h; ih++){
		for(size_t iw=0; iw<in_w; iw++){
			w.window_idx.push_back(size);

			for(size_t kh=0; kh<kernel; kh++){
				ptrdiff_t const oh = static_cast<ptrdiff_t>(ih * stride + kh * dilation) - static_cast<ptrdiff_t>(padding);
				if(oh < 0 || static_cast<size_t>(oh) >= out_h) continue;

				for(size_t kw=0; kw<kernel; kw++){
					ptrdiff_t const ow = static_cast<ptrdiff_t>(iw * stride + kw * dilation) - static_cast<ptrdiff_t>(padding);
					if(ow < 0 || static_cast<size_t>(ow) >= out_w) continue;

					w.map_window.push_back(static_cast<size_t>(oh) * out_w + static_cast<size_t>(ow));	// delta pixel
					w.kernel_window.push_back(kh * kernel + kw);										// kernel position
					size++;
				}
			}
		}
	}

	w.window_idx.push_back(size);
	w.map_window.shrink_to_fit();
	w.kernel_window.shrink_to_fit();
	w.window_idx.shrink_to_fit();
	w.initialized = true;
}

// GRAD-WEIGHT: per kernel position, the pairs (input pixel, delta pixel).
// dweight[kh,kw] += sum input[ih,iw]*delta[oh,ow]  with  oh = ih*stride-pad+kh*dil
inline void build_transposed_grad_weight_window(Window& w,
                                                size_t const in_h, size_t const in_w,
                                                size_t const out_h, size_t const out_w,
                                                size_t const kernel, size_t const stride,
                                                size_t const padding, size_t const dilation) {
	w.output_height = kernel;
	w.output_width  = kernel;
	w.map_window.clear();
	w.kernel_window.clear();
	w.window_idx.clear();
	w.window_idx.reserve(kernel * kernel + 1);

	size_t size = 0;

	for(size_t kh=0; kh<kernel; kh++){
		for(size_t kw=0; kw<kernel; kw++){
			w.window_idx.push_back(size);

			for(size_t ih=0; ih<in_h; ih++){
				ptrdiff_t const oh = static_cast<ptrdiff_t>(ih * stride + kh * dilation) - static_cast<ptrdiff_t>(padding);
				if(oh < 0 || static_cast<size_t>(oh) >= out_h) continue;

				for(size_t iw=0; iw<in_w; iw++){
					ptrdiff_t const ow = static_cast<ptrdiff_t>(iw * stride + kw * dilation) - static_cast<ptrdiff_t>(padding);
					if(ow < 0 || static_cast<size_t>(ow) >= out_w) continue;

					w.map_window.push_back(ih * in_w + iw);								// input pixel
					w.kernel_window.push_back(static_cast<size_t>(oh) * out_w + static_cast<size_t>(ow));	// delta pixel
					size++;
				}
			}
		}
	}

	w.window_idx.push_back(size);
	w.map_window.shrink_to_fit();
	w.kernel_window.shrink_to_fit();
	w.window_idx.shrink_to_fit();
	w.initialized = true;
}

// ----------------------------------------------------------------------------
//  FORWARD
// ----------------------------------------------------------------------------
template <size_t nbits, size_t es>
void transposed_convolution2d_thread(StdTensor<posit<nbits, es>> const& input,
                                     StdTensor<posit<nbits, es>> const& weight,
                                     StdTensor<posit<nbits, es>> const& bias,
                                     StdTensor<posit<nbits, es>>& output,
                                     Window const* w,
                                     size_t input_batch, size_t output_batch,
                                     size_t const n_samples) {

	bool const no_bias = bias.empty();
	size_t const in_channels  = weight.shape()[0];	// weight: [in_c, out_c, k, k]
	size_t const out_channels = weight.shape()[1];

	size_t const input_batch_stride   = input.strides()[0];
	size_t const input_channel_stride = input.strides()[1];
	size_t const output_batch_stride  = output.strides()[0];
	size_t const output_channel_stride= output.strides()[1];	// = out_h * out_w
	size_t const weight_in_channel_stride  = weight.strides()[0];
	size_t const weight_out_channel_stride = weight.strides()[1];

	size_t const out_size = output_channel_stride;

	Quire<nbits, es> q;

	for(size_t s=0; s<n_samples; s++){
		size_t output_channel = output_batch;
		size_t weight_out_channel = 0;

		// Loop over output channels
		for(size_t oc=0; oc<out_channels; oc++){

			// Loop over output pixels
			for(size_t o=0; o<out_size; o++){
				if(no_bias)	q.clear();
				else		q = bias[oc];

				size_t input_channel = input_batch;
				size_t weight_in_channel = weight_out_channel;

				// Accumulate all input channels into a single quire (one rounding)
				for(size_t ic=0; ic<in_channels; ic++){
					do_convolution(input, weight, q, *w, input_channel, weight_in_channel, o);
					input_channel    += input_channel_stride;
					weight_in_channel += weight_in_channel_stride;
				}

				convert(q.to_value(), output[output_channel + o]);
			}

			output_channel     += output_channel_stride;
			weight_out_channel += weight_out_channel_stride;
		}

		input_batch  += input_batch_stride;
		output_batch += output_batch_stride;
	}
}

template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> transposed_convolution2d(StdTensor<posit<nbits, es>> const& input,
                                                     StdTensor<posit<nbits, es>> const& weight,
                                                     StdTensor<posit<nbits, es>> const& bias,
                                                     size_t const stride=1,
                                                     size_t const padding=0,
                                                     size_t const output_padding_h=0,
                                                     size_t const output_padding_w=0,
                                                     size_t const dilation=1,
                                                     Window* w=nullptr) {

	size_t const batch_size   = input.shape()[0];
	size_t const out_channels = weight.shape()[1];
	size_t const kernel       = weight.shape()[2];	// square

	// output_padding is per axis. The kernel, stride and padding are square, as
	// everywhere else in the framework, but the two axes still need separate
	// output_padding whenever the input is not: Conv2d::backward has to undo a
	// floor division that can drop a row without dropping a column.
	size_t const in_h  = input.shape()[2];
	size_t const in_w  = input.shape()[3];
	size_t const out_h = transposed_output_size(in_h, kernel, stride, padding, output_padding_h, dilation);
	size_t const out_w = transposed_output_size(in_w, kernel, stride, padding, output_padding_w, dilation);

	bool const owns = (w == nullptr);
	if(owns)
		w = new Window();
	if(!w->initialized)
		build_transposed_forward_window(*w, in_h, in_w, out_h, out_w, kernel, stride, padding, dilation);

	StdTensor<posit<nbits, es>> output({batch_size, out_channels, out_h, out_w});

	size_t const input_batch_stride  = input.strides()[0];
	size_t const output_batch_stride = output.strides()[0];

#ifdef USING_LL_THREADS

	const size_t max_threads = (LL_THREADS < batch_size) ? LL_THREADS : batch_size;
	std::vector<std::thread> threads;
	threads.reserve(max_threads);

	size_t const n_samples      = batch_size / max_threads;
	size_t const nthreads_more  = batch_size % max_threads;

	size_t input_samples_begin  = 0;
	size_t output_samples_begin = 0;

	for(size_t t=0; t<max_threads; t++){
		size_t const thread_samples = n_samples + (t < nthreads_more ? 1 : 0);

		threads.push_back(std::thread(transposed_convolution2d_thread<nbits, es>,
		                              std::cref(input), std::cref(weight), std::cref(bias), std::ref(output), w,
		                              input_samples_begin, output_samples_begin, thread_samples));

		input_samples_begin  += thread_samples * input_batch_stride;
		output_samples_begin += thread_samples * output_batch_stride;
	}

	for(std::thread& t : threads)
		t.join();

#else

	transposed_convolution2d_thread<nbits, es>(input, weight, bias, output, w,
	                                           0, 0, batch_size);

#endif

	if(owns)
		delete w;

	return output;
}

// ----------------------------------------------------------------------------
//  GRAD-INPUT
// ----------------------------------------------------------------------------
template <size_t nbits, size_t es>
void transposed_convolution2d_backward_input_thread(StdTensor<posit<nbits, es>> const& grad_output,
                                                    StdTensor<posit<nbits, es>> const& weight,
                                                    StdTensor<posit<nbits, es>>& grad_input,
                                                    Window const* w,
                                                    size_t batch_begin, size_t const n_samples) {

	size_t const in_channels  = grad_input.shape()[1];
	size_t const out_channels = grad_output.shape()[1];

	size_t const grad_out_batch_stride   = grad_output.strides()[0];
	size_t const grad_out_channel_stride = grad_output.strides()[1];
	size_t const grad_in_batch_stride    = grad_input.strides()[0];
	size_t const grad_in_channel_stride  = grad_input.strides()[1];	// = in_h * in_w
	size_t const weight_in_channel_stride  = weight.strides()[0];
	size_t const weight_out_channel_stride = weight.strides()[1];

	size_t const in_size = grad_in_channel_stride;

	size_t grad_in_batch  = batch_begin * grad_in_batch_stride;
	size_t grad_out_batch = batch_begin * grad_out_batch_stride;

	Quire<nbits, es> q;

	for(size_t s=0; s<n_samples; s++){
		size_t grad_in_channel = grad_in_batch;
		size_t weight_in_channel = 0;

		// Loop over input channels
		for(size_t ic=0; ic<in_channels; ic++){

			// Loop over input pixels
			for(size_t i=0; i<in_size; i++){
				q.clear();

				size_t grad_out_channel = grad_out_batch;
				size_t weight_out_channel = weight_in_channel;

				// Accumulate all output channels into a single quire
				for(size_t oc=0; oc<out_channels; oc++){
					do_convolution(grad_output, weight, q, *w, grad_out_channel, weight_out_channel, i);
					grad_out_channel   += grad_out_channel_stride;
					weight_out_channel += weight_out_channel_stride;
				}

				convert(q.to_value(), grad_input[grad_in_channel + i]);
			}

			grad_in_channel   += grad_in_channel_stride;
			weight_in_channel += weight_in_channel_stride;
		}

		grad_in_batch  += grad_in_batch_stride;
		grad_out_batch += grad_out_batch_stride;
	}
}

template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> transposed_convolution2d_backward_input(StdTensor<posit<nbits, es>> const& grad_output,
                                                                    StdTensor<posit<nbits, es>> const& weight,
                                                                    size_t const in_height, size_t const in_width,
                                                                    size_t const stride=1,
                                                                    size_t const padding=0,
                                                                    size_t const dilation=1,
                                                                    Window* w=nullptr) {

	size_t const batch_size  = grad_output.shape()[0];
	size_t const in_channels = weight.shape()[0];	// weight: [in_c, out_c, k, k]
	size_t const kernel      = weight.shape()[2];

	size_t const out_h = grad_output.shape()[2];
	size_t const out_w = grad_output.shape()[3];

	bool const owns = (w == nullptr);
	if(owns)
		w = new Window();
	if(!w->initialized)
		build_transposed_grad_input_window(*w, in_height, in_width, out_h, out_w, kernel, stride, padding, dilation);

	StdTensor<posit<nbits, es>> grad_input({batch_size, in_channels, in_height, in_width});

#ifdef USING_LL_THREADS

	const size_t max_threads = (LL_THREADS < batch_size) ? LL_THREADS : batch_size;
	std::vector<std::thread> threads;
	threads.reserve(max_threads);

	size_t const n_samples     = batch_size / max_threads;
	size_t const nthreads_more = batch_size % max_threads;

	size_t batch_begin = 0;

	for(size_t t=0; t<max_threads; t++){
		size_t const thread_samples = n_samples + (t < nthreads_more ? 1 : 0);

		threads.push_back(std::thread(transposed_convolution2d_backward_input_thread<nbits, es>,
		                              std::cref(grad_output), std::cref(weight), std::ref(grad_input), w,
		                              batch_begin, thread_samples));

		batch_begin += thread_samples;
	}

	for(std::thread& t : threads)
		t.join();

#else

	transposed_convolution2d_backward_input_thread<nbits, es>(grad_output, weight, grad_input, w,
	                                                          0, batch_size);

#endif

	if(owns)
		delete w;

	return grad_input;
}

// ----------------------------------------------------------------------------
//  GRAD-WEIGHT
// ----------------------------------------------------------------------------
template <size_t nbits, size_t es>
void transposed_convolution2d_gradient_thread(StdTensor<posit<nbits, es>> const& input,
                                              StdTensor<posit<nbits, es>> const& delta,
                                              StdTensor<posit<nbits, es>>& dweight,
                                              Window const* w,
                                              size_t const dweight_begin, size_t const n_samples) {

	size_t const batch_size   = input.shape()[0];
	size_t const in_channels  = input.shape()[1];
	size_t const out_channels = delta.shape()[1];

	size_t const input_batch_stride   = input.strides()[0];
	size_t const input_channel_stride = input.strides()[1];
	size_t const delta_batch_stride   = delta.strides()[0];
	size_t const delta_channel_stride = delta.strides()[1];

	size_t const ksize = dweight.strides()[1];	// kernel * kernel

	// dweight: [in_c, out_c, k, k]. Split the start index into (ic, oc, kernel pos).
	size_t const ic0   = dweight_begin / dweight.strides()[0];
	size_t const oc0   = (dweight_begin % dweight.strides()[0]) / dweight.strides()[1];
	size_t const kpos0 = dweight_begin % dweight.strides()[1];

	Quire<nbits, es> q;
	size_t n = dweight_begin;
	size_t const n_end = dweight_begin + n_samples;
	bool first = true;

	for(size_t ic=ic0; ic<in_channels; ic++){
		size_t const input_channel0 = ic * input_channel_stride;

		for(size_t oc=(first ? oc0 : 0); oc<out_channels; oc++){
			size_t const delta_channel0 = oc * delta_channel_stride;

			for(size_t kpos=(first ? kpos0 : 0); kpos<ksize; kpos++){
				q.clear();

				size_t input_base = input_channel0;
				size_t delta_base = delta_channel0;

				// Accumulate over the whole batch into a single quire
				for(size_t b=0; b<batch_size; b++){
					do_convolution(input, delta, q, *w, input_base, delta_base, kpos);
					input_base += input_batch_stride;
					delta_base += delta_batch_stride;
				}

				convert(q.to_value(), dweight[n++]);

				if(n == n_end)
					return;

				first = false;
			}
		}
	}
}

template <size_t nbits, size_t es>
StdTensor<posit<nbits, es>> transposed_convolution2d_gradient(StdTensor<posit<nbits, es>> const& input,
                                                              StdTensor<posit<nbits, es>> const& delta,
                                                              size_t const kernel,
                                                              size_t const stride=1,
                                                              size_t const padding=0,
                                                              size_t const dilation=1,
                                                              Window* w=nullptr) {

	size_t const in_channels  = input.shape()[1];
	size_t const out_channels = delta.shape()[1];

	size_t const in_h  = input.shape()[2];
	size_t const in_w  = input.shape()[3];
	size_t const out_h = delta.shape()[2];
	size_t const out_w = delta.shape()[3];

	bool const owns = (w == nullptr);
	if(owns)
		w = new Window();
	if(!w->initialized)
		build_transposed_grad_weight_window(*w, in_h, in_w, out_h, out_w, kernel, stride, padding, dilation);

	StdTensor<posit<nbits, es>> dweight({in_channels, out_channels, kernel, kernel});
	size_t const size_total = dweight.size();

#ifdef USING_LL_THREADS

	const size_t max_threads = (LL_THREADS < size_total) ? LL_THREADS : size_total;
	std::vector<std::thread> threads;
	threads.reserve(max_threads);

	size_t const n_samples     = size_total / max_threads;
	size_t const nthreads_more = size_total % max_threads;

	size_t dweight_begin = 0;

	for(size_t t=0; t<max_threads; t++){
		size_t const thread_samples = n_samples + (t < nthreads_more ? 1 : 0);

		threads.push_back(std::thread(transposed_convolution2d_gradient_thread<nbits, es>,
		                              std::cref(input), std::cref(delta), std::ref(dweight), w,
		                              dweight_begin, thread_samples));

		dweight_begin += thread_samples;
	}

	for(std::thread& t : threads)
		t.join();

#else

	transposed_convolution2d_gradient_thread<nbits, es>(input, delta, dweight, w,
	                                                    0, size_total);

#endif

	if(owns)
		delete w;

	return dweight;
}

#endif /* TRANSPOSED_CONVOLUTION_HPP */

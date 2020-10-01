#ifndef WINDOW_HPP
#define WINDOW_HPP

// General headers
#include <vector>

// Custom headers
#include "StdTensor.hpp"

// Get windows/indices for convolution and pooling operations
struct Window{
	// Flag to know when it is initialized
	bool initialized = false;

	// Output dimensions
	size_t output_height, output_width;

	// Window from output to input and from output to kernel (weights)
	std::vector<size_t> map_window;
	std::vector<size_t> kernel_window;

	// Indices of above vectors where window starts for entry of output
	std::vector<size_t> window_idx;

	// Used when you know the start and want to go forward
	void output_to_input(	size_t const input_height0, size_t const input_width0,
							size_t const kernel_height, size_t const kernel_width,
							size_t const stride=1, size_t const padding=0,
							size_t const dilation_input=1, size_t dilation_weight=1	){

		StdTensor<size_t> input_idx = sequence<size_t>({input_height0, input_width0});
		if(dilation_input>1)
			input_idx = dilate<size_t>(input_idx, dilation_input, -1);
		if(padding>0)
			input_idx = pad<size_t>(input_idx, padding, -1);

		size_t const input_height = input_idx.shape()[0];
		size_t const input_width = input_idx.shape()[1];
	
		// Calculate output dimensions
		output_height = (input_height - kernel_height)/stride + 1;
		output_width = (input_width - kernel_width)/stride + 1;
		
		//output_height = (input_height + dilation*(input_height-1) + 2*padding - kernel_height)/stride + 1;
		//output_width = (input_width + dilation*(input_width-1) + 2*padding - kernel_width)/stride + 1;

		// Clear window vectors
		map_window.clear();
		kernel_window.clear();
		window_idx.clear();

		// Reserve sizes for vectors
		map_window.reserve(output_height * output_width * kernel_height * kernel_width);
		kernel_window.reserve(output_height * output_width * kernel_height * kernel_width);
		window_idx.reserve(output_height * output_width + 1);		// plus one to store size of vector

		size_t const iend = input_height-kernel_height;
		size_t const jend = input_width-kernel_width;
		size_t const None = static_cast<size_t>(-1);
		size_t size = 0;

		// OUTPUT -> INPUT - MAP
		// Loop through rows of input
		for(size_t i=0; i<=iend; i+=stride){
			// Loop through columns of input
			for(size_t j=0; j<=jend; j+=stride){
				// Register where indices begin
				window_idx.push_back(size);

				// Loop through rows of kernel overlapped with input
				for(size_t m=i, x=0; x<kernel_height; m++, x++){
					// Loop through columns of kernel overlapped with input
					for(size_t n=j, y=0; y<kernel_width; n++, y++){

						// Push indices of input elements that overlap with kernel
						size_t index = m*input_width+n;
						if(input_idx[index] != None) {
							map_window.push_back(input_idx[index]);
							kernel_window.push_back(x*kernel_width+y);
							size++;
						}
					}
				}
			}	
		}
		
		// Add size of window in order to know when to "stop"
		window_idx.push_back(size);

		// Deallocate unnecessary space
		map_window.shrink_to_fit();
		kernel_window.shrink_to_fit();
		window_idx.shrink_to_fit();

		// Windows are initialized
		initialized = true;

		return;
	}

	// Used when you know the end and want to go back
	void input_to_output(	size_t const input_height0, size_t const input_width0,
							size_t const kernel_height, size_t const kernel_width,
							size_t const stride, size_t const padding	){

		StdTensor<size_t> input_idx = sequence<size_t>({input_height0, input_width0});
		if(padding>0)
			input_idx = pad<size_t>(input_idx, padding, -1);

		size_t const input_height = input_idx.shape()[0];
		size_t const input_width = input_idx.shape()[1];
	
		// Calculate output dimensions
		output_height = (input_height - kernel_height)/stride + 1;
		output_width = (input_width - kernel_width)/stride + 1;

		//output_height = (input_height + 2*padding - kernel_height)/stride + 1;
		//output_width = (input_width + 2*padding - kernel_width)/stride + 1;

		// Clear window vectors
		map_window.clear();
		kernel_window.clear();
		window_idx.clear();

		// Reserve sizes for vectors
		map_window.reserve(output_height * output_width * kernel_height * kernel_width);
		kernel_window.reserve(output_height * output_width * kernel_height * kernel_width);
		window_idx.reserve(output_height * output_width + 1);		// plus one to store size of vector

		std::vector<std::vector<size_t>> temp_map(input_height0 * input_width0);
		std::vector<std::vector<size_t>> temp_kernel(input_height0 * input_width0);

		size_t const iend = input_height-kernel_height;
		size_t const jend = input_width-kernel_width;
		size_t const None = static_cast<size_t>(-1);
		size_t output_idx = 0;

		// INPUT -> OUTPUT - MAP
		// Loop through rows of input
		for(size_t i=0; i<=iend; i+=stride){
			// Loop through columns of input
			for(size_t j=0; j<=jend; j+=stride){

				// Loop through rows of kernel overlapped with input
				for(size_t m=i, x=0; x<kernel_height; m++, x++){
					// Loop through columns of kernel overlapped with input
					for(size_t n=j, y=0; y<kernel_width; n++, y++){

						// Push indices of input elements that overlap with kernel
						size_t index = m*input_width+n;
						if(input_idx[index] != None) {
							temp_map[input_idx[index]].push_back(output_idx);
							temp_kernel[x*kernel_width+y].push_back(output_idx);
						}
					}
				}
				output_idx++;
			}	
		}

		// Convert temp structure to desired format (window and idx)
		// Calculate size of temp
		size_t size = std::accumulate(temp_map.begin(), temp_map.end(), 0,
			[](size_t sum, std::vector<size_t> const& indices){ return sum + indices.size(); });

		map_window.reserve(size);
		kernel_window.reserve(size);
		size_t idx = 0;

		for(size_t i=0, size=temp_map.size(); i<size; i++){
			window_idx.push_back(idx);
			idx += temp_map[i].size();

			std::move(temp_map[i].begin(), temp_map[i].end(), std::back_inserter(map_window));
			std::move(temp_kernel[i].begin(), temp_kernel[i].end(), std::back_inserter(kernel_window));
		}

		// Add size of window in order to know when to "stop"
		window_idx.push_back(size);
		
		// Windows are initialized
		initialized = true;

		return;
	}
};

#endif /* WINDOW_HPP */

#ifndef WINDOW_HPP
#define WINDOW_HPP

// General headers
#include <vector>

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

	void forward(	int const input_height, int const input_width,
					int const kernel_height, int const kernel_width,
					int const stride, int const padding	){
	
		// Calculate output dimensions
		output_height = (input_height + 2*padding - kernel_height)/stride + 1;
		output_width = (input_width + 2*padding - kernel_width)/stride + 1;

		// Clear window vectors
		map_window.clear();
		kernel_window.clear();
		window_idx.clear();

		// Reserve sizes for vectors
		map_window.reserve(output_height * output_width * kernel_height * kernel_width);
		kernel_window.reserve(output_height * output_width * kernel_height * kernel_width);
		window_idx.reserve(output_height * output_width + 1);		// plus one to store size of vector

		size_t size = 0;

		// OUTPUT -> INPUT - MAP
		// Loop through rows of input
		for(int i=0-padding; i<=input_height+padding-kernel_height; i+=stride){
			// Loop through columns of input
			for(int j=0-padding; j<=input_width+padding-kernel_width; j+=stride){
				// Register where indices begin
				window_idx.push_back(size);

				// Loop through rows of kernel overlapped with input
				for(int m=i; m<i+kernel_height; m++){
					// Loop through columns of kernel overlapped with input
					for(int n=j; n<j+kernel_width; n++){
						
						// Push indices of input elements that overlap with kernel
						if(m>=0 && m<input_height && n>=0 && n<input_width) {
							map_window.push_back(m*input_width+n);
							kernel_window.push_back((m-i)*kernel_width+(n-j));
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

	void backward(	int const input_height, int const input_width,
					int const kernel_height, int const kernel_width,
					int const stride, int const padding	){
	
		// Calculate output dimensions
		output_height = (input_height + 2*padding - kernel_height)/stride + 1;
		output_width = (input_width + 2*padding - kernel_width)/stride + 1;

		// Clear window vectors
		map_window.clear();
		kernel_window.clear();
		window_idx.clear();

		// Reserve sizes for vectors
		map_window.reserve(output_height * output_width * kernel_height * kernel_width);
		kernel_window.reserve(output_height * output_width * kernel_height * kernel_width);
		window_idx.reserve(output_height * output_width + 1);		// plus one to store size of vector

		std::vector<std::vector<size_t>> temp_map(input_height * input_width);
		std::vector<std::vector<size_t>> temp_kernel(input_height * input_width);
		size_t output_idx = 0;

		// INPUT -> OUTPUT - MAP
		// Loop through rows of input
		for(int i=0-padding; i<=input_height+padding-kernel_height; i+=stride){
			// Loop through columns of input
			for(int j=0-padding; j<=input_width+padding-kernel_width; j+=stride){
				// Loop through rows of kernel overlapped with input
				for(int m=i; m<i+kernel_height; m++){
					// Loop through columns of kernel overlapped with input
					for(int n=j; n<j+kernel_width; n++){
						
						// Push indices of output element when input overlaps with kernel
						if(m>=0 && m<input_height && n>=0 && n<input_width) {
							temp_map[m*input_width+n].push_back(output_idx);
							temp_kernel[(m-i)*kernel_width+(n-j)].push_back(output_idx);
						}
					}
				}
				output_idx++;
			}	
		}

		// Convert temp structure to desired format (window and idx)
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

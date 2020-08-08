#ifndef LOAD_DATA_HPP
#define LOAD_DATA_HPP

// General headers
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector> 

void load_data(	const std::string& input_path,
				const std::string& output_path,
				std::vector<std::vector<float>>& input,
				std::vector<float>& output ){
// Receives input and output samples paths
// Receives two vectors of where to store the input and output data
// Input will be a vector and output a single value

	using namespace std;

	string line;
	size_t idx;
	float n;
	size_t n_lines, n_columns;

	cout << "Loading input data from " << input_path << endl;

	// Open input file
	ifstream in_file(input_path);

	if (in_file.is_open()) {
		// Count number of lines
		n_lines = std::count(	std::istreambuf_iterator<char>(in_file), 
        		     			std::istreambuf_iterator<char>(), '\n');	
		in_file.seekg (0);	// return to file beginning
		
		// Count # of numbers per line
		getline(in_file, line);
		n_columns = std::count(line.begin(), line.end(), ' ') + 1;
		in_file.seekg (0);	// return to file beginning

		// Resize vector of samples
		input.resize(n_lines);

		// Loop through lines and extract columns to make up an input vector
    	for (size_t i=0; i<n_lines; i++){ 
			getline(in_file, line);

			input[i].resize(n_columns);

			idx = 0;	// start at the first float
			for (size_t j=0; j<n_columns; j++){
				line = line.substr(idx);	// get the float after idx
				n = std::stof(line, &idx);	// parse float and point to next float
				input[i][j] = n;			// store float in input vector
			}
    	}
		in_file.close();
	}
	else cout << "Unable to open file"; 

	cout << "Loading output data from " << output_path << endl;

	// Open output file
	ifstream out_file(output_path);

	if (out_file.is_open()) {
		// Count number of lines
		n_lines = std::count(	std::istreambuf_iterator<char>(out_file), 
        		     			std::istreambuf_iterator<char>(), '\n');	
		out_file.seekg (0);	// return to file beginning
		
		// Resize vector of samples
		output.resize(n_lines);

		// Loop through lines and extract value
    	for (size_t i=0; i<n_lines; i++){ 
			getline(out_file, line);
			n = std::stof(line, &idx);	// parse float
			output[i] = n;				// store float as output
    	}
		out_file.close();
	}
	else cout << "Unable to open file"; 

	cout << "input size = " << input.size() << " output size = " << output.size() << endl;

	return;
}

#endif /* LOAD_DATA_HPP */

#ifndef CUSTOMDATASET_HPP
#define CUSTOMDATASET_HPP

// General headers
#include <torch/torch.h>

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
	// Declare 2 vectors of tensors for images and labels
	std::vector<torch::Tensor> x, y;
public:
	// Constructor
	CustomDataset(std::vector<std::vector<float>>& x_data, std::vector<float>& y_data) {
	
		x.resize(x_data.size());
		long long size_vector = x_data[0].size();

		for(std::vector<int>::size_type i = 0; i < x_data.size(); i++){
			x[i] = torch::from_blob(x_data[i].data(), {size_vector}).clone();
		}

		y.resize(y_data.size());
		for(std::vector<int>::size_type i = 0; i < y_data.size(); i++)
			y[i] = torch::full({1}, y_data[i]);
	}

	// Override get() function to return tensor at location index
	torch::data::Example<> get(size_t index) override {
		torch::Tensor sample_x = x.at(index);
		torch::Tensor sample_y = y.at(index);
		return {sample_x.clone(), sample_y.clone()};
	}

	// Return the length of data
	torch::optional<size_t> size() const override {
		return y.size();
	}
};

#endif /* CUSTOMDATASET_HPP */

#ifndef TRAIN_FLOAT_HPP
#define TRAIN_FLOAT_HPP

// General headers
#include <iostream>
#include <torch/torch.h>

template <class Model, typename DataLoader>
void train_float(	size_t epoch,
					size_t const num_epochs,
					Model& model,
					DataLoader& data_loader,
					torch::optim::Optimizer& optimizer,
					size_t const kLogInterval,
					size_t const dataset_size	){

	model->train();
	size_t batch_idx = 0;
	size_t total_batch_size = 0;

	for(auto& batch : data_loader) {
		// Update number of trained samples
		batch_idx++;
		total_batch_size += batch.target.size(0);
		
		// Get data and target
		auto data = batch.data;
		auto target = batch.target;
		
		// Convert data and target to float32 and long
		data = data.to(torch::kF32);
		target = target.to(torch::kLong);

		// Forward pass
		auto output = model->forward(data);
		auto loss = torch::nn::functional::nll_loss(output, target);

		// Backward pass and optimize
		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		// Print progress
		if (batch_idx % kLogInterval == 0) {
			std::printf("Train Epoch: %.3f/%2ld Data: %5ld/%5ld Loss: %.4f\n",
					epoch-1+static_cast<float>(total_batch_size)/dataset_size,
					num_epochs,
					total_batch_size,
					dataset_size,
					loss.template item<float>());
		}
	}		
}

#endif /* TRAIN_FLOAT_HPP */

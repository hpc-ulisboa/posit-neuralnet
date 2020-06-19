#ifndef TRAIN_POSIT_HPP
#define TRAIN_POSIT_HPP

// General headers
#include <iostream>
#include <torch/torch.h>
#include <positnn/positnn>

// Custom headers
#include "PositNet.hpp"

template <typename Posit, typename DataLoader, typename Optimizer>
void train_posit(	size_t epoch,
					size_t const num_epochs,
					PositNet<Posit>& model,
					DataLoader& data_loader,
					Optimizer& optimizer,
					size_t const kLogInterval,
					size_t const dataset_size	){

	model.train();
	size_t batch_idx = 0;
	size_t total_batch_size = 0;

	for(auto& batch : data_loader) {
		// Update number of trained samples
		batch_idx++;
		size_t const batch_size = batch.target.size(0);
		total_batch_size += batch_size;

		// Get data and target
		auto data_float = batch.data;
		auto target_float = batch.target;

		// Flatten data
		data_float = data_float.reshape({batch.data.size(0), 784});

		// Convert data and target to float32 and long
		data_float = data_float.to(torch::kF32);
		target_float = target_float.to(torch::kLong);

		// Convert data and target from PyTorch Tensor to StdTensor
		auto data = Tensor_to_StdTensor<float, Posit>(data_float);
		auto target = Tensor_to_StdTensor<long, long>(target_float);
		
#ifndef USING_HL_THREADS
		// Forward pass
		auto output = model.forward(data);
		cross_entropy_loss<Posit, long> loss(output, target);
		
		// Backward pass and optimize
		optimizer.zero_grad();
		loss.backward(model);
		optimizer.step();
#else
		// Forward and backward pass
		std::vector<float> losses;
		std::vector<std::vector<StdTensor<Posit>>> gradients;
		forward_backward<cross_entropy_loss<Posit, long>>(model, data, target, losses, gradients);

		// Sum gradients from other threads
		optimizer.zero_grad();
		sum_gradients(model.parameters(), gradients);
	
		// Optimizer step
		optimizer.step();
#endif /* USING_HL_THREADS */

		// Print progress
		if (batch_idx % kLogInterval == 0) {
#ifndef USING_HL_THREADS
			float loss_value = loss.template item<float>();
#else
			float loss_value = std::accumulate(losses.begin(), losses.end(), 0.0) / batch_size;
#endif /* USING_HL_THREADS */

			std::printf("Train Epoch: %.3f/%2ld Data: %5ld/%5ld Loss: %.4f\n",
					epoch-1+static_cast<float>(total_batch_size)/dataset_size,
					num_epochs,
					total_batch_size,
					dataset_size,
					loss_value);
		}
	}
}

#endif /* TRAIN_POSIT_HPP */

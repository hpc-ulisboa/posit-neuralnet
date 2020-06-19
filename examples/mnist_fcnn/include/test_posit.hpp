#ifndef TEST_POSIT_HPP
#define TEST_POSIT_HPP

// General headers
#include <iostream>
#include <torch/torch.h>
#include <positnn/positnn>

// Custom headers
#include "PositNet.hpp"

template <typename Posit, typename DataLoader>
void test_posit(	PositNet<Posit>& model,
					DataLoader& data_loader,
					size_t dataset_size	){
	
	model.eval();
	float test_loss = 0;
	size_t correct = 0;

	for(const auto& batch : data_loader) {
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

		// Forward pass
		auto output = model.forward(data);

		// Calculate loss
		test_loss += cross_entropy_loss<Posit, long>(	output,
														target,
														Reduction::Sum	).template item<float>();
	
		auto pred = output.template argmax<long>(1);
		correct += pred.eq(target).template sum<size_t>();
	}

	// Get average loss
	test_loss /= dataset_size;

	// Print results
	std::printf("Test set: Loss: %.4f | Accuracy: %.4f\n",
	  			test_loss,
	  			static_cast<float>(correct) / dataset_size);
}

#endif /* TEST_POSIT_HPP */

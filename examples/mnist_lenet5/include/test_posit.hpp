#ifndef TEST_POSIT_HPP
#define TEST_POSIT_HPP

// General headers
#include <iostream>
#include <torch/torch.h>
#include <positnn/positnn>

template <typename T, template<typename> class Model, typename DataLoader>
void test_posit(	Model<T>& model,
					DataLoader& data_loader,
					size_t dataset_size	){
	
	using Target = unsigned char;

	model.eval();
	float test_loss = 0;
	size_t correct = 0;

	for(auto const& batch : data_loader) {
		// Get data and target
		auto data_float = batch.data;
		auto target_float = batch.target;

		// Convert data and target to float32 and uint8
		data_float = data_float.to(torch::kF32);
		target_float = target_float.to(torch::kUInt8);

		// Convert data and target from PyTorch Tensor to StdTensor
		auto data = Tensor_to_StdTensor<float, typename T::Forward>(data_float);
		auto target = Tensor_to_StdTensor<uint8_t, Target>(target_float);

#ifndef USING_HL_THREADS
		// Forward pass
		auto output = model.forward(data);
		
		// Calculate loss
		test_loss += cross_entropy_loss<typename T::Loss, typename T::Loss, Target>(
						output,
						target,
						Reduction::Sum	).template item<float>();

		// Get prediction from output
		auto pred = output.template argmax<Target>(1);
#else
		// Forward pass and prediction
		std::vector<float> losses;
		auto pred = forward<cross_entropy_loss<typename T::Loss, typename T::Loss, Target>>(
						model, data, target, losses);

		// Calculate loss
		test_loss += std::accumulate(losses.begin(), losses.end(), 0.0);
#endif /* USING_HL_THREADS */

		correct += pred.eq(target).template sum<size_t>();
	}

	// Get average loss
	test_loss /= dataset_size;

	// Print results
	std::printf("Test set: Loss: %.4f | Accuracy: [%5ld/%5ld] %.4f\n",
	  			test_loss, correct, dataset_size,
	  			static_cast<float>(correct) / dataset_size);
}

#endif /* TEST_POSIT_HPP */

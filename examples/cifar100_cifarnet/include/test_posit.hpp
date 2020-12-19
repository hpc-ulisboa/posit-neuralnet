// General headers
#include <cstdint>
#include <iostream>
#include <torch/torch.h>
#include <positnn/positnn>

template <typename Type, template<typename> class Model, typename DataLoader>
void test_posit(Model<Type>& model, DataLoader& data_loader, size_t dataset_size) {
	// Setup inference
	model.eval();
	float test_loss = 0;
	size_t correct = 0;
	size_t correct_top5 = 0;
	
	// Setup data types
	using F = typename Type::Forward;
	using L = typename Type::Loss;
	using T = unsigned short int;
	
	// Loop the entire testing dataset
	for(auto const& batch : data_loader) {
		// Get data and target
		auto data_float = batch.data;
		auto target_float = batch.target;
		
		// Convert data and target to float32 and uint8
		data_float = data_float.to(torch::kF32);
		target_float = target_float.to(torch::kUInt8);
		
		// Convert data and target from PyTorch Tensor to StdTensor
		auto data = Tensor_to_StdTensor<float, F>(data_float);
		auto target = Tensor_to_StdTensor<uint8_t, T>(target_float);
		
		// Forward pass
		auto output = model.forward(data);
		
		// Calculate loss
		test_loss += cross_entropy_loss<L>(output, target,
						Reduction::Sum).template item<float>();
		
		// Get prediction from output
		auto pred = output.template argmax<T>(1);
		auto top5 = output.template topk<T>(5);
		correct_top5 += target.in(top5).template sum<size_t>();
		correct += pred.eq(target).template sum<size_t>();
	}
	
	// Get average loss
	test_loss /= dataset_size;
	
	// Print results
	std::printf("Test set: Loss: %.4f | Accuracy: [%5ld/%5ld] %.4f | Top-5: [%5ld/%5ld] %.4f\n",
	  			test_loss,
				correct, dataset_size, static_cast<float>(correct) / dataset_size,
				correct_top5, dataset_size, static_cast<float>(correct_top5) / dataset_size);
}

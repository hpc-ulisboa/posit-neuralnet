// General headers
#include <iostream>
#include <stdio.h>
#include <torch/torch.h>

// Custom headers
#include "LeNet5_float.hpp"

// Custom functions
#include "test_float.hpp"
#include "train_float.hpp"

// Dataset path
#define DATASET_PATH				"../dataset"

// Load
#define NET_LOAD_FILENAME_FLOAT		"../net/example/model_epoch_0_float.pt"

// Save
#define NET_SAVE_PATH				"../net/example/"
#define NET_EPOCH_FILENAME_FLOAT	"model_epoch_%zu_float.pt"

// Options
#define LOAD false
#define SAVE_UNTRAINED true
#define SAVE_EPOCH true

template<class ModelFloat>
void save_model(std::string save_path, ModelFloat& model_float, size_t const epoch) {
	// Float
	char net_epoch_filename_float[128];
	snprintf(net_epoch_filename_float, sizeof(net_epoch_filename_float),
			NET_EPOCH_FILENAME_FLOAT, epoch);
	save_path += net_epoch_filename_float;

	torch::save(model_float, save_path);
}
		
int main() {
	// Line buffering
	setvbuf(stdout, NULL, _IOLBF, 0);

    std::cout << "Fashion MNIST Classification" << std::endl;
    std::cout << "Training and Testing on CPU." << std::endl;
	if(SAVE_UNTRAINED || SAVE_EPOCH)
		std::cout << "Save path: " << NET_SAVE_PATH << std::endl;
	
	// Training and Testing settings
	// The batch size for training.
	size_t const kTrainBatchSize = 64;

	// The batch size for testing.
	size_t const kTestBatchSize = 1024;
	
	// The number of epochs to train.
	size_t const num_epochs = 10;

	// After how many batches to log a new update with the loss value.
	size_t const kLogInterval = 32;

	// Optimizer parameters
	float learning_rate = 1./16;
	float const momentum = 0.5;
	size_t const adaptive_lr = 4;

	// Float networks
	LeNet5_float model_float;

	// Load net parameters from file
	if(LOAD){
		torch::load(model_float, NET_LOAD_FILENAME_FLOAT);
	}

	// Save net before training
	if(SAVE_UNTRAINED)
		save_model(NET_SAVE_PATH, model_float, 0);

	// Load Fashion MNIST training dataset
	auto train_dataset = torch::data::datasets::MNIST(DATASET_PATH)
							.map(torch::data::transforms::Normalize<>(0.2860, 0.3300))
                    		.map(torch::data::transforms::Stack<>());
	const size_t train_dataset_size = train_dataset.size().value();

	// Create data loader from training dataset
	auto train_loader = torch::data::make_data_loader(
			std::move(train_dataset),
			torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));

	// Load Fashion MNIST testing dataset
	auto test_dataset = torch::data::datasets::MNIST(DATASET_PATH,
							torch::data::datasets::MNIST::Mode::kTest)
							.map(torch::data::transforms::Normalize<>(0.2860, 0.3300))
                    		.map(torch::data::transforms::Stack<>());
	const size_t test_dataset_size = test_dataset.size().value();

	// Create data loader from testing dataset
	auto test_loader = torch::data::make_data_loader(
							std::move(test_dataset),
							torch::data::DataLoaderOptions().batch_size(kTestBatchSize));

	// Optimizer
    torch::optim::SGD optimizer_float(model_float->parameters(), torch::optim::SGDOptions(learning_rate).momentum(momentum));

	// Test with untrained models
	// Float
	std::cout << std::endl << "Float" << std::endl;
	test_float(model_float, *test_loader, test_dataset_size);
	
    // Train the model
    std::cout << std::endl << "Running..." << std::endl;
    for (size_t epoch = 1; epoch<=num_epochs; epoch++) {
		train_float(epoch, num_epochs, model_float, *train_loader, optimizer_float, kLogInterval, train_dataset_size);
		test_float(model_float, *test_loader, test_dataset_size);
		
		// Save models after each epoch
		if(SAVE_EPOCH)
			save_model(NET_SAVE_PATH, model_float, epoch);
		
		// Update learning rate every adaptive_lr epochs
		if(adaptive_lr>0 && epoch%adaptive_lr==0) {
			learning_rate /= 2.;
			static_cast<torch::optim::SGDOptions &>(optimizer_float.param_groups()[0].options()).lr(learning_rate);
		}
    }

    std::cout << "Finished!\n";

	return 0;
}

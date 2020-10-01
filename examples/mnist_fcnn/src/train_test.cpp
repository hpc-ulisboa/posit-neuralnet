// General headers
#include <iostream>
#include <torch/torch.h>
#include <universal/posit/posit>
#include <positnn/positnn>
#include <stdio.h>

// Custom headers
#include "FloatNet.hpp"
#include "PositNet.hpp"

// Custom functions
#include "test_float.hpp"
#include "test_posit.hpp"
#include "train_float.hpp"
#include "train_posit.hpp"

// Namespaces
using namespace sw::unum;

// Posit configuration
struct Type{
	typedef posit<16, 1> Optimizer;
	typedef posit<8, 2> Forward;
	typedef posit<8, 2> Backward;
	typedef posit<8, 2> Gradient;
	typedef posit<16, 1> Loss;
	typedef posit<16, 1> SaveFile;
	typedef posit<16, 1> LoadFile;
};

// Dataset path
#define DATASET_PATH					"../dataset"

// Load
#define NET_LOAD_FILENAME_FLOAT			"../net/posit_8_1/model_epoch_0_float.pt"
#define NET_LOAD_FILENAME_POSIT			"../net/posit_8_2/model_epoch_0_posit.dat"

// Save
#define NET_SAVE_PATH					"../net/example/"
#define NET_EPOCH_FILENAME_FLOAT		NET_SAVE_PATH "model_epoch_%zu_float.pt"
#define NET_EPOCH_FILENAME_POSIT		NET_SAVE_PATH "model_epoch_%zu_posit.dat"

// Options
#define LOAD false
#define COPY true
#define SAVE_UNTRAINED false
#define SAVE_EPOCH false

template <typename T>
void save_models(size_t const epoch, FloatNet& model_float, PositNet<T>& model_posit) {
	char net_epoch_filename_float[128];
	char net_epoch_filename_posit[128];
	// Float
	snprintf(net_epoch_filename_float, sizeof(net_epoch_filename_float),
			NET_EPOCH_FILENAME_FLOAT, epoch);
	torch::save(model_float, net_epoch_filename_float);

	// Posit
	snprintf(net_epoch_filename_posit, sizeof(net_epoch_filename_posit),
			NET_EPOCH_FILENAME_POSIT, epoch);
	save<T>(model_posit, net_epoch_filename_posit);
}
		
int main() {
	// Line buffering
	setvbuf(stdout, NULL, _IOLBF, 0);

	std::cout << "MNIST Classification" << std::endl;
    std::cout << "Training and Testing on CPU." << std::endl;
    std::cout << "PositOptimizer<" << Type::Optimizer::nbits << ", " << Type::Optimizer::es << ">" << std::endl;
    std::cout << "PositForward<" << Type::Forward::nbits << ", " << Type::Forward::es << ">" << std::endl;
    std::cout << "PositBackward<" << Type::Backward::nbits << ", " << Type::Backward::es << ">" << std::endl;
    std::cout << "PositGradient<" << Type::Gradient::nbits << ", " << Type::Gradient::es << ">" << std::endl;
    std::cout << "PositLoss<" << Type::Loss::nbits << ", " << Type::Loss::es << ">" << std::endl;
	
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
	float const learning_rate = 0.1;
	float const momentum = 0.5;
	 
	// Start training posit at epoch
	size_t const start_epoch = 1;

	// Float and Posit networks
    FloatNet model_float;
	PositNet<Type> model_posit;

	// Load net parameters from file
	if(LOAD){
		torch::load(model_float, NET_LOAD_FILENAME_FLOAT);
		if(!COPY)
			load<PositLoadFile>(model_posit, NET_LOAD_FILENAME_POSIT);
	}

	// Initialize posit net with the same random parameters as float net
	if(COPY)
		copy_parameters(model_float->parameters(), model_posit.parameters());

	// Save net before training
	if(SAVE_UNTRAINED)
		save_models(0, model_float, model_posit);

	// Load MNIST training dataset
	auto train_dataset = torch::data::datasets::MNIST(DATASET_PATH)
							.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                    		.map(torch::data::transforms::Stack<>());
	const size_t train_dataset_size = train_dataset.size().value();

	// Create data loader from training dataset
	auto train_loader = torch::data::make_data_loader(
	//auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
							std::move(train_dataset),
							torch::data::DataLoaderOptions().batch_size(kTrainBatchSize));
	
	// Load MNIST testing dataset
	auto test_dataset = torch::data::datasets::MNIST(DATASET_PATH,
							torch::data::datasets::MNIST::Mode::kTest)
							.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                    		.map(torch::data::transforms::Stack<>());
	const size_t test_dataset_size = test_dataset.size().value();

	// Create data loader from testing dataset
	auto test_loader = torch::data::make_data_loader(
	//auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
							std::move(test_dataset),
							torch::data::DataLoaderOptions().batch_size(kTestBatchSize));

	// Optimizer
    torch::optim::SGD optimizer_float(model_float->parameters(), torch::optim::SGDOptions(learning_rate).momentum(momentum));
    SGD<Type::Optimizer> optimizer_posit(model_posit.parameters(), SGDOptions<Type::Optimizer>(learning_rate, momentum));

	// Test with untrained models
	// Float
	std::cout << std::endl << "Float" << std::endl;
	test_float(model_float, *test_loader, test_dataset_size);
	
	//Posit
	std::cout << std::endl << "Posit" << std::endl;
	test_posit(model_posit, *test_loader, test_dataset_size);

    // Train the model
    std::cout << "Running...\n";
    for (size_t epoch = 1; epoch<=num_epochs; ++epoch) {
		// Float
		std::cout << std::endl << "Float" << std::endl;
		train_float(epoch, num_epochs, model_float, *train_loader, optimizer_float, kLogInterval, train_dataset_size);
		test_float(model_float, *test_loader, test_dataset_size);
		
		// Posit
		std::cout << std::endl << "Posit" << std::endl;
		if(epoch<start_epoch) {
			copy_parameters(model_float->parameters(), model_posit.parameters());
		}
		else {
			train_posit(epoch, num_epochs, model_posit, *train_loader, optimizer_posit, kLogInterval, train_dataset_size);
		}
		test_posit(model_posit, *test_loader, test_dataset_size);

		// Save models after each epoch
		if(SAVE_EPOCH)
			save_models(epoch, model_float, model_posit);
    }

    std::cout << "Finished!\n";

	return 0;
}

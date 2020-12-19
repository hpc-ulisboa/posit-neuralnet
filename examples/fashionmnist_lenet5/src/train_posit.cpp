// General headers
#include <iostream>
#include <torch/torch.h>
#include <universal/posit/posit>
#include <positnn/positnn>
#include <stdio.h>

// Custom headers
#include "LeNet5_float.hpp"
#include "LeNet5_posit.hpp"

// Custom functions
#include "test_posit.hpp"
#include "train_posit.hpp"
#include "train_float.hpp"

// Namespaces
using namespace sw::unum;

// Posit configuration
struct Type{
	typedef posit<16, 2> Optimizer;
	typedef posit<8, 2> Forward;
	typedef posit<8, 2> Backward;
	typedef posit<8, 2> Gradient;
	typedef posit<16, 2> Loss;
	typedef Optimizer SaveFile;
	typedef posit<16, 2> LoadFile;
};

// Dataset path
#define DATASET_PATH				"../dataset"

// Load
#define NET_LOAD_FILENAME_FLOAT		"../net/example/model_epoch_0_float.pt"
#define NET_LOAD_FILENAME_POSIT		"../net/example/model_epoch_0_posit.dat"

// Save
#define NET_SAVE_PATH				"../net/example/"
#define NET_EPOCH_FILENAME_POSIT	"model_epoch_%zu_posit.dat"

// Options
#define LOAD false
#define COPY true
#define SAVE_UNTRAINED true
#define SAVE_EPOCH true

template<typename T, template<typename> class ModelPosit>
void save_model(std::string save_path, ModelPosit<T>& model_posit, size_t const epoch) {
	// Posit
	char net_epoch_filename_posit[128];
	snprintf(net_epoch_filename_posit, sizeof(net_epoch_filename_posit),
			NET_EPOCH_FILENAME_POSIT, epoch);
	save_path += net_epoch_filename_posit;

	save<typename T::SaveFile>(model_posit, save_path);
}
	
int main() {
	// Line buffering
	setvbuf(stdout, NULL, _IOLBF, 0);
	
    std::cout << "Fashion MNIST Classification" << std::endl;
    std::cout << "Training and Testing on CPU." << std::endl;
	std::cout << "Underflow mode: " << UNDERFLOW_MODE << std::endl;
	std::cout << "Quire mode: " << QUIRE_MODE << std::endl;
    std::cout << "PositOptimizer<" << Type::Optimizer::nbits << ", " << Type::Optimizer::es << ">" << std::endl;
    std::cout << "PositForward<" << Type::Forward::nbits << ", " << Type::Forward::es << ">" << std::endl;
    std::cout << "PositBackward<" << Type::Backward::nbits << ", " << Type::Backward::es << ">" << std::endl;
    std::cout << "PositGradient<" << Type::Gradient::nbits << ", " << Type::Gradient::es << ">" << std::endl;
    std::cout << "PositLoss<" << Type::Loss::nbits << ", " << Type::Loss::es << ">" << std::endl;
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

	// Float and Posit networks
	LeNet5_float model_float;
	LeNet5_posit<Type> model_posit;

	// Load net parameters from file
	if(LOAD){
		if(COPY)
			torch::load(model_float, NET_LOAD_FILENAME_FLOAT);
		else
			load<Type::LoadFile>(model_posit, NET_LOAD_FILENAME_POSIT);
	}

	// Initialize posit net with the same random parameters as float net
	if(COPY)
		copy_parameters(model_float->parameters(), model_posit.parameters());

	// Save net before training
	if(SAVE_UNTRAINED)
		save_model(NET_SAVE_PATH, model_posit, 0);

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
    SGD<Type::Optimizer> optimizer_posit(model_posit.parameters(), SGDOptions<Type::Optimizer>(learning_rate, momentum));

	// Test with untrained models
	//Posit
	std::cout << std::endl << "Posit" << std::endl;
	test_posit(model_posit, *test_loader, test_dataset_size);

    // Train the model
    std::cout << std::endl << "Running..." << std::endl;
    for (size_t epoch = 1; epoch<=num_epochs; ++epoch) {
		train_posit(epoch, num_epochs, model_posit, *train_loader, optimizer_posit, kLogInterval, train_dataset_size);
		test_posit(model_posit, *test_loader, test_dataset_size);
		
		// Save models after each epoch
		if(SAVE_EPOCH)
			save_model(NET_SAVE_PATH, model_posit, epoch);
		
		// Update learning rate every adaptive_lr epochs
		if(adaptive_lr>0 && epoch%adaptive_lr==0) {
			learning_rate /= 2.;
			static_cast<torch::optim::SGDOptions &>(optimizer_float.param_groups()[0].options()).lr(learning_rate);
			optimizer_posit.options().learning_rate = learning_rate;
		}
    }

    std::cout << "Finished!\n";

	return 0;
}

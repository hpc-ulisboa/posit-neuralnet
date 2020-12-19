#ifndef CIFARNET_FLOAT_HPP
#define CIFARNET_FLOAT_HPP

// General headers
#include <torch/torch.h>

struct CifarNet_floatImpl : torch::nn::Module {
	CifarNet_floatImpl(size_t num_classes=100) :
		conv1(torch::nn::Conv2dOptions(3, 8, /*kernel_size=*/5).padding(2)),
		conv2(torch::nn::Conv2dOptions(8, 16, /*kernel_size=*/5).padding(2)),
		fc1(1024, 384),
		fc2(384, 192),
		fc3(192, num_classes)
	{ 
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("fc1", fc1);
		register_module("fc2", fc2);
		register_module("fc3", fc3);
	}

	torch::Tensor forward(torch::Tensor x) {
		// Convolutional layers
		x = torch::relu(torch::max_pool2d(conv1->forward(x), /*kernel_size=*/2, /*stride=*/2));
		x = torch::relu(torch::max_pool2d(conv2->forward(x), /*kernel_size=*/2, /*stride=*/2));

		// Flatten
		x = x.view({-1, 1024});
		
		// Fully connected layers
		x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
		x = torch::relu(fc1->forward(x));
    	x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
		x = torch::relu(fc2->forward(x));
		x = fc3->forward(x);

		return torch::log_softmax(x, /*dim=*/ 1);
	}

	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
	torch::nn::Linear fc3;
};

TORCH_MODULE(CifarNet_float);

#endif /* CIFARNET_FLOAT_HPP */

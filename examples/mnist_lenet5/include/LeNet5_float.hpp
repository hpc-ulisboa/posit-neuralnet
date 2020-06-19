#ifndef LENET5_FLOAT_HPP
#define LENET5_FLOAT_HPP

// General headers
#include <torch/torch.h>

struct LeNet5_floatImpl : torch::nn::Module {
	LeNet5_floatImpl() :
		//conv1(torch::nn::Conv2dOptions(1, 6, /*kernel_size=*/5)),
		conv1(torch::nn::Conv2dOptions(1, 6, /*kernel_size=*/5).padding(2)),
		conv2(torch::nn::Conv2dOptions(6, 16, /*kernel_size=*/5)),
		conv3(torch::nn::Conv2dOptions(16, 120, /*kernel_size=*/5)),
		fc1(120, 84),
		fc2(84, 10)
	{ 
		register_module("conv1", conv1);
		register_module("conv2", conv2);
		register_module("conv3", conv3);
		register_module("fc1", fc1);
		register_module("fc2", fc2);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = torch::relu(torch::max_pool2d(conv1->forward(x), /*kernel_size=*/2, /*stride=*/2));
		x = torch::relu(torch::max_pool2d(conv2->forward(x), /*kernel_size=*/2, /*stride=*/2));
		x = torch::relu(conv3->forward(x));
		x = x.view({-1, 120});
		x = torch::relu(fc1->forward(x));
		x = fc2->forward(x);
		return torch::log_softmax(x, /*dim=*/ 1);
	}

	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Conv2d conv3;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
};

TORCH_MODULE(LeNet5_float);

#endif /* LENET5_FLOAT_HPP */

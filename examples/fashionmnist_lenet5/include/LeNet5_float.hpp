// General headers
#include <torch/torch.h>

struct LeNet5_floatImpl : torch::nn::Module {
	LeNet5_floatImpl() :
		conv1(torch::nn::Conv2dOptions(1, 6, 5).padding(2)),
		conv2(torch::nn::Conv2dOptions(6, 16, 5)),
		conv3(torch::nn::Conv2dOptions(16, 120, 5)),
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
		x = conv1->forward(x);
		x = torch::max_pool2d(x, 2, 2);
		x = torch::relu(x);
		
		x = conv2->forward(x);
		x = torch::max_pool2d(x, 2, 2);
		x = torch::relu(x);
		
		x = conv3->forward(x);
		x = torch::relu(x);
		
		x = x.view({-1, 120});
		
		x = fc1->forward(x);
		x = torch::relu(x);
		
		x = fc2->forward(x);
		return torch::log_softmax(x, 1);
	}
	
	torch::nn::Conv2d conv1, conv2, conv3;
	torch::nn::Linear fc1, fc2;
};

TORCH_MODULE(LeNet5_float);

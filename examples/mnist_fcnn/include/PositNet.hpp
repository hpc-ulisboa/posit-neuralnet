#ifndef NET_HPP
#define NET_HPP

// Custom headers
#include <positnn/positnn>

template <typename T>
class PositNet : public Layer<T>{
public:
	PositNet() :
		linear1(784, 32),
		linear2(32, 10)
	{
		this->register_module(linear1);
		this->register_module(linear2);
	}

	StdTensor<T> forward(StdTensor<T> x) {
		// Flatten data
		x.reshape({x.shape()[0], 784});

		x = linear1.forward(x);
		x = relu.forward(x);

		x = linear2.forward(x);
		return x;
	}

	StdTensor<T> backward(StdTensor<T> x) {
		x = linear2.backward(x);
		
		x = relu.backward(x);
		x = linear1.backward(x);
		return x;
	}

private:
	Linear<T> linear1, linear2;
	ReLU relu;
};

#endif /* NET_HPP */

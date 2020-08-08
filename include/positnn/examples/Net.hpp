#ifndef NET_HPP
#define NET_HPP

// General headers
#include <positnn/positnn>

template <typename Posit>
class Net : public Layer<Posit>{
public:
	Net(size_t in, size_t hidden, size_t out) :
		linear1(in, hidden),
		linear2(hidden, out)
	{
		this->register_module(linear1);
		this->register_module(linear2);
	}

	StdTensor<Posit> forward(StdTensor<Posit>& input) {
		StdTensor<Posit> output = linear1.forward(input);
		output = sigmoid1.forward(output);
		return linear2.forward(output);
	}

	StdTensor<Posit> backward(StdTensor<Posit>& output_error) {
		StdTensor<Posit> error = linear2.backward(output_error);
		error = sigmoid1.backward(error);
		return linear1.backward(error);
	}

private:
	Linear<Posit> linear1;
	Sigmoid<Posit> sigmoid1;
	Linear<Posit> linear2;
};

// activation receives left term and multiplies by sigma'(z_x,l), returns delta
// linear receives delta and multiplies by weight and returns that (left term)
// linear also calculates the gradient

#endif /* NET_HPP */

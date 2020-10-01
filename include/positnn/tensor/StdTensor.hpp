// Mostly based from Konrad Rudolph's answer in
// https://stackoverflow.com/questions/47664127/create-a-multidimensional-array-dynamically-in-c

#ifndef STDTENSOR_HPP
#define STDTENSOR_HPP

// General headers
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

// Custom headers
#include "../utils/type_name.hpp"
#include "../utils/utils.hpp"

template <typename T>
class StdTensor{
public:
	StdTensor()	:
		m_dim(0),
		m_size(0)
	{ }

	StdTensor(const std::vector<size_t>& shape0) : 
		m_dim(shape0.size()),
		m_size(std::accumulate(shape0.begin(), shape0.end(),
			1, std::multiplies<size_t>())),
		m_shape(shape0),
		m_data(m_size)
	{
		compute_strides();
	}

	StdTensor(const size_t size0) : 
		m_dim(1),
		m_size(size0),
		m_shape({size0}),
		m_strides({1}),
		m_data(size0)
	{ }

	/*
	StdTensor(const std::vector<size_t>& shape0, const bool reserve) :
		m_dim (shape0.size()),
		m_size (std::accumulate(shape0.begin(), shape0.end(),
			1, std::multiplies<size_t>())),
		m_shape (shape0)
	{
		m_data.reserve(m_size);
		compute_strides();
	}
	*/

	// TODO: constructor that receives data and shape
	// TODO: constructor that receives shape and value (all entries are the same)

	~StdTensor() { }

	// Default constructors and assignment operators
	constexpr StdTensor(StdTensor<T> const&) = default;
	constexpr StdTensor(StdTensor<T>&&) = default;

	StdTensor<T>& operator=(StdTensor<T> const&) = default;
	StdTensor<T>& operator=(StdTensor<T>&&) = default;

	// Assignment operator when StdTensors have different types
	template <class otherT>
	//StdTensor<T>& operator=(StdTensor<otherT> const& rhs) {
	StdTensor(StdTensor<otherT> const& rhs) {
		m_dim = rhs.dim();
		m_size = rhs.size();
		m_shape = rhs.shape();
		m_strides = rhs.strides();

		const size_t old_size = m_data.size();
		size_t copy_size;

		if(old_size > m_size) {
			m_data.resize(m_size);
			m_data.shrink_to_fit();
			copy_size = m_size;
		}
		else {
			m_data.reserve(m_size);
			copy_size = old_size;
		}

		for(size_t i=0; i<copy_size; i++){
			//if(m_data[i] != rhs[i])
				m_data[i] = T(rhs[i]);
		}

		for(size_t i=copy_size; i<m_size; i++)
			m_data.push_back(T(rhs[i]));

		//return *this;
	}

	// Get element from tensor
	//typename std::vector<T>::reference const operator[](size_t i) const {
	const T& operator[](size_t i) const {
		return m_data[i];
	}

	T& operator[](size_t i) {
		return m_data[i];
	}

	const T& operator[](const std::vector<size_t>& indices) const {
		auto flat_index = std::inner_product(
				indices.begin(), indices.end(),
				m_strides.begin(), 0);
		return m_data[flat_index];
	}

	T& operator[](const std::vector<size_t>& indices) {
		auto flat_index = std::inner_product(
				indices.begin(), indices.end(),
				m_strides.begin(), 0);
		return m_data[flat_index];
	}

	// Reshape tensor
	void reshape(const std::vector<size_t>& new_shape) {
		size_t new_size = std::accumulate(
							new_shape.begin(), new_shape.end(),
							1, std::multiplies<size_t>());
		
		// If tensor has different size, tensor will be resized
		if (new_size != m_size) {
			m_data.resize(new_size);
			m_size = new_size;
		}

		m_dim = new_shape.size();
		m_shape = new_shape;
		compute_strides();
	}

	// Get shape
	const std::vector<size_t>& shape() const {
		return m_shape;
	}

	// Get strides
	const std::vector<size_t>& strides() const {
		return m_strides;
	}

	// Get dimension
	size_t dim() const {
		return m_dim;
	}
	
	// Get size
	size_t size() const {
		return m_size;
	}	

	// Get a reference to the vector with the data
	std::vector<T>& vector() {
		return m_data;
	}

	// Get a reference to the vector with the data
	const std::vector<T>& vector() const {
		return m_data;
	}

	// Get a pointer to the data
	const T* data() const {
		return m_data.data();
	}	

	// Iterators
	typedef typename std::vector<T>::iterator iterator;
	typedef typename std::vector<T>::const_iterator const_iterator;

	iterator begin() {
		return m_data.begin();
	}	
	
	iterator end() {
		return m_data.end();
	}

	const_iterator begin() const{
		return m_data.begin();
	}	
	
	const_iterator end() const {
		return m_data.end();
	}	

	bool empty() const {
		return m_data.empty();
	}

	// Set tensor to zero
	void clear() {
		for(size_t i=0; i<m_size; i++)
			m_data[i].clear();

		return;
	}

	// Set tensor with a value
	void set(const T& value) {
		for(size_t i=0; i<m_size; i++)
			m_data[i] = value;

		return;
	}

	template <typename otherT>
	void set(const otherT& value) {
		T aux = T(value);
		for(size_t i=0; i<m_size; i++)
			m_data[i] = aux;

		return;
	}

	// Slice vector along first dimension
	StdTensor<T> slice(size_t const begin, size_t const end) const{
		// TODO: throw error if begin>=end
		std::vector<size_t> new_shape = m_shape;
		new_shape[0] = end-begin;

		size_t const stride = m_strides[0];

		StdTensor<T> output(new_shape);
		output.vector() = std::vector<T>(m_data.begin()+begin*stride, m_data.begin()+end*stride);

		return output;
	}

	/*
	void push_back(const T& elem) {
		m_data.push_back(elem);
	}
	*/

	// Wrote posit to file (binary)
	template <typename PositFile=T>
	void write(std::ostream& out) {
		out.write((char*)&m_dim, sizeof(m_dim));
		out.write((char*)&m_size, sizeof(m_size));
		write_vector(out, m_shape);
		write_vector(out, m_strides);
		write_vector_posit<T, PositFile>(out, m_data);
	}

	// Read posit from file (binary)
	template <typename PositFile=T>
	void read(std::istream& in) {
		in.read((char*)&m_dim, sizeof(m_dim));
		in.read((char*)&m_size, sizeof(m_size));
		read_vector(in, m_shape);
		read_vector(in, m_strides);
		read_vector_posit<T, PositFile>(in, m_data);
	}
	
	// Print operator
	friend std::ostream& operator<< (std::ostream &out, const StdTensor& tensor){
		out << tensor.m_data << std::endl;
		out << "[ " << type_name<T>() << '{' << tensor.m_shape << "} ]";
		// TODO: Improve print to have multiple lines like a matrix
		return out;
	}

	// Assignment of arithmetic operators
	// Addition operator
	StdTensor<T>& operator+= (const StdTensor<T>& other){
		size_t i;
		const size_t other_size = other.size();

		for(i=0; i<m_size; i++){
			m_data[i] += other[i%other_size];	// TODO: WARN THAT B IS REPEATED
		}

		return *this;
	}

	template <class otherT>
	StdTensor<T>& operator+= (const StdTensor<otherT>& other){
		return *this += StdTensor<T>(other);
	}

	template <class Scalar>
	StdTensor<T>& operator+= (const Scalar& other){
		size_t i;
		const T aux = T(other);

		for(i=0; i<m_size; i++){
			m_data[i] += aux;
		}

		return *this;
	}

	// Subtraction operator
	StdTensor<T>& operator-= (const StdTensor<T>& other){
		size_t i;
		const size_t other_size = other.size();

		for(i=0; i<m_size; i++){
			m_data[i] -= other[i%other_size];	// TODO: WARN THAT B IS REPEATED
		}

		return *this;
	}

	template <typename Scalar>
	StdTensor<T>& operator-= (const Scalar& other){
		size_t i;
		const T aux = T(other);

		for(i=0; i<m_size; i++){
			m_data[i] -= aux;
		}

		return *this;
	}

	// Multiplication operator
	StdTensor<T>& operator*= (const StdTensor<T>& other){
		size_t i;
		const size_t other_size = other.size();

		for(i=0; i<m_size; i++){
			m_data[i] *= other[i%other_size];	// TODO: WARN THAT B IS REPEATED
		}

		return *this;
	}

	template <typename Scalar>
	StdTensor<T>& operator*= (const Scalar& other){
		size_t i;
		const T aux = T(other);

		for(i=0; i<m_size; i++){
			m_data[i] *= aux;
		}

		return *this;
	}
	
	// Division operator
	StdTensor<T>& operator/= (const StdTensor<T>& other){
		size_t i;
		const size_t other_size = other.size();

		for(i=0; i<m_size; i++){
			m_data[i] /= other[i%other_size];	// TODO: WARN THAT B IS REPEATED
		}

		return *this;
	}	

	template <typename Scalar>
	StdTensor<T>& operator/= (const Scalar& other){
		size_t i;
		const T aux = T(other);

		for(i=0; i<m_size; i++){
			m_data[i] /= aux;
		}

		return *this;
	}

	// Compare two StdTensors for equality. Uses char instead of bool due to:
	// https://gitlab.dune-project.org/core/dune-common/issues/19
	// To fix, substitute declaration of operator[]
	template <typename otherT>
	StdTensor<unsigned char> eq (const StdTensor<otherT>& other) const {
		StdTensor<unsigned char> result(m_shape);

		for(size_t i=0; i<m_size; i++) {
			if(otherT(m_data[i]) == other[i]) {
				result[i] = 1;
			}
		}

		return result;
	}
	
	// Argmax along axis
	template <typename argT=size_t>
	StdTensor<argT> argmax(const size_t axis=0){
		std::vector<size_t> new_shape = m_shape;
		new_shape.erase(new_shape.begin()+axis);
		if (new_shape.empty())
			new_shape.push_back(1);

		StdTensor<argT> argMaxTensor(new_shape);

		size_t const axis_size = m_shape[axis];
		size_t const stride = m_strides[axis];
		size_t const loop_stride = axis_size*stride;

		argT index;
		T max;

		for(size_t i=0, n=0; i<m_size; i+=loop_stride) {	// loop blocks
			for(size_t j=0; j<stride; j++){	// loop beginning elements of block
				index = 0;
				max = m_data[i+j];
				for(size_t k=i+j+stride, l=1; l<axis_size; k+=stride, l++){	// loop elements to sum
					if(m_data[k] > max) {
						index = l;
						max = m_data[k];
					}
				}
				argMaxTensor[n++] = index;
			}
		}

		return argMaxTensor;
	}

	// Sum of all elements of tensor. To use quires in the sum, use matrix::sum() instead.
	template <typename sumT=T>
	sumT sum(){
		sumT result = 0;

		for(size_t i=0; i<m_size; i++) {
			result += sumT(m_data[i]);
		}

		return result;
	}	

private:
	void compute_strides() {
		m_strides.resize(m_dim);
		m_strides[m_dim - 1] = 1;
		std::partial_sum(m_shape.rbegin(),
				m_shape.rend() - 1,
				m_strides.rbegin() + 1,
				std::multiplies<size_t>());
	}

	size_t m_dim;
	size_t m_size;
	std::vector<size_t> m_shape;
	std::vector<size_t> m_strides;
	std::vector<T> m_data;
	
//public:
	//std::vector<T> m_data;
};

// Binary arithmetic operators
// Addition operator
template <typename T>
inline StdTensor<T> operator+ (const StdTensor<T>& a, const StdTensor<T>& b) {
	StdTensor<T> c = a;
	c += b;
	return c;
}

// Subtraction operator
template <typename T>
inline StdTensor<T> operator- (const StdTensor<T>& a, const StdTensor<T>& b) {
	StdTensor<T> c = a;
	c -= b;
	return c;
}

// Multiplication operator
template <typename T>
inline StdTensor<T> operator* (const StdTensor<T>& a, const StdTensor<T>& b) {
	StdTensor<T> c = a;
	c *= b;
	return c;
}

template <typename T, typename Scalar>
inline StdTensor<T> operator* (const StdTensor<T>& a, const Scalar& b) {
	StdTensor<T> c = a;
	c *= b;
	return c;
}

// Division operator
template <typename T>
inline StdTensor<T> operator/ (const StdTensor<T>& a, const StdTensor<T>& b) {
	StdTensor<T> c = a;
	c /= b;
	return c;
}

template <typename T, typename Scalar>
inline StdTensor<T> operator/ (const StdTensor<T>& a, const Scalar& b) {
	StdTensor<T> c = a;
	c /= b;
	return c;
}

// Functions to apply to tensors (useful in convolution layer)

template <typename T>
StdTensor<T> sequence(std::vector<size_t> const& shape) {
	StdTensor<T> y(shape);

	for(size_t i=0, size=y.size(); i<size; i++) {
		y[i] = T(i);
	}

	return y;
}

template <typename T>
StdTensor<T> pad(StdTensor<T> const& x, unsigned int const padding, T const value=0) {
	if(padding==0)
		return x;

	int const height0 = x.shape()[0];
	int const width0 = x.shape()[1];

	size_t const height = height0 + 2*padding;
	size_t const width = width0 + 2*padding;

	StdTensor<T> y({height, width});
	int const iend = height0 + padding;
	int const jend = width0 + padding;
	size_t m=0, n=0;

	// Loop through rows
	for(int i=-padding; i<iend; i++) {
		if(i<0 || i>=height0) {
			// Pad top and bottom rows
			for(int j=-padding; j<jend; j++) {
				y[m++] = value;
			}
		}
		else {
			// Loop through columns
			for(int j=-padding, jend=width0+padding; j<jend; j++) {
				if(j<0 || j>=width0) {
					y[m++] = value;
				}
				else {
					y[m++] = x[n++];
				}
			}
		}
	}

	return y;
}

template <typename T>
StdTensor<T> dilate(StdTensor<T> const& x, size_t const dilation, T const value=0) {
	if(dilation==1)
		return x;

	size_t const height = (x.shape()[0]-1)*dilation + 1;
	size_t const width = (x.shape()[1]-1)*dilation + 1;

	StdTensor<T> y({height, width});
	size_t const expand = dilation-1;
	size_t const stride = expand*width;
	size_t const jend = width-1;
	size_t k = 0;

	// Loop through rows
	for(size_t i=0, size=y.size(); i<size; ) {
		// Loop through cols
		for(size_t j=0; j<jend; j+=dilation) {
			// Copy value to dilated
			y[i++] = x[k++];
			//std::cerr << "copy\t y[" << i-1 << "] = " << y[i-1] << std::endl;
			
			// Assign value to dilated cols
			for(size_t l=0; l<expand; l++) {
				y[i++] = value;
				//std::cerr << "dilate\t y[" << i-1 << "] = " << y[i-1] << std::endl;
			}
		}
		// Copy value of last column
		y[i++] = x[k++];
		//std::cerr << "last\t y[" << i-1 << "] = " << y[i-1] << std::endl;

		if(i<size) {
			// Assign value to dilated rows
			for(size_t l=0; l<stride; l++){
				y[i++] = value;
				//std::cerr << "row\t y[" << i-1 << "] = " << y[i-1] << std::endl;
			}
		}
	}

	return y;
}

#endif /* STDTENSOR_HPP */

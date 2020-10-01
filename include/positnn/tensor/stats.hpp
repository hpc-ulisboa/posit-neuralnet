#ifndef STATS_HPP
#define STATS_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "StdTensor.hpp"
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

// Mean of tensor
template <typename T> 
T calculate_mean(StdTensor<T> const& x) {
	size_t const size = x.size();
	Quire<T::nbits, T::es> sum;
	sum.clear();
	T mean;

	// Calculate mean
	for(size_t i=0; i<size; i++){
		sum += x[i];
	}

	convert(sum.to_value(), mean);
	mean /= size;

	return mean;
}

// Variance of tensor
template <typename T> 
T calculate_var(StdTensor<T> const& x, size_t ddof=0) {
	// Calculate mean
	T mean = calculate_mean(x);

	size_t const size = x.size();
	Quire<T::nbits, T::es> sum;
	sum.clear();
	T var;

	// Calculate variance
	for(size_t i=0; i<size; i++){
		T delta = mean - x[i];
		sum += Quire_mul(delta, delta);
	}

	convert(sum.to_value(), var);
	var /= (size-ddof);

	return var;
}

// Standard deviation of tensor
template <typename T> 
T calculate_std(StdTensor<T> const& x, size_t ddof=0) {
	// Calculate variance
	T std = calculate_var(x, ddof);
	std = sqrt(std);

	return std;
}

#endif /* STATS_HPP */

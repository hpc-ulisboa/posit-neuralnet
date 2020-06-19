#ifndef STATS_HPP
#define STATS_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "StdTensor.hpp"

// Namespaces
using namespace sw::unum;

// Mean of tensor
template <typename Positi, typename Positf=Positi> 
Positf calculate_mean(StdTensor<Positi> const& x) {
	size_t const size = x.size();
	quire<Positf::nbits, Positf::es, Positf::nbits-1> sum = 0;
	Positf mean;

	// Calculate mean
	for(size_t i=0; i<size; i++){
		sum += x[i];
	}

	convert(sum.to_value(), mean);
	mean /= size;

	return mean;
}

// Variance of tensor
template <typename Positi, typename Positf=Positi> 
Positf calculate_var(StdTensor<Positi> const& x) {
	// Calculate mean
	Positf mean = calculate_mean<Positi, Positf>(x);

	size_t const size = x.size();
	quire<Positf::nbits, Positf::es, Positf::nbits-1> sum = 0;
	Positf var;

	// Calculate variance
	for(size_t i=0; i<size; i++){
		Positf delta = mean;
		delta -= x[i];
		sum += quire_mul(delta, delta);
	}

	convert(sum.to_value(), var);
	var /= size;

	return var;
}

// Standard deviation of tensor
template <typename Positi, typename Positf=Positi> 
Positf calculate_std(StdTensor<Positi> const& x) {
	// Calculate variance
	Positf std = calculate_var<Positi, Positf>(x);
	std = sqrt(std);

	return std;
}

#endif /* STATS_HPP */

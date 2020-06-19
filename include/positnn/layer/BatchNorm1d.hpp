#ifndef BATCHNORM1D_HPP
#define BATCHNORM1D_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "Layer.hpp"
#include "../tensor/matrix.hpp"
#include "../tensor/sum.hpp"
#include "../tensor/StdTensor.hpp"

// Namespaces
using namespace sw::unum;

template <typename Posit>
class BatchNorm1d : public Layer<Posit> {
public:
	BatchNorm1d(size_t _num_features, Posit _eps=1e-5, Posit _momentum=0.1,
				bool _affine=true, bool _track_running_stats=true) :
		num_features(_num_features),
		eps(_eps), momentum(_momentum),
		affine(_affine), track_running_stats(_track_running_stats),
		gamma(_num_features), beta(_num_features),   
		gamma_gradient(_num_features), beta_gradient(_num_features),
		stddev(_num_features)
		
	{
		if(track_running_stats){
			running_mean = StdTensor<Posit>(num_features);
			running_variance = StdTensor<Posit>(num_features);
		}

		this->register_parameter(gamma, gamma_gradient);
		this->register_parameter(beta, beta_gradient);

		reset_parameters();
	}

	void reset_parameters() {
		gamma.set(Posit(1));
		beta.set(Posit(0));

		// TODO: confirm initialization
		running_mean.set(Posit(0));
		running_variance.set(Posit(1));
	}

	StdTensor<Posit> forward(StdTensor<Posit>& x) {
		StdTensor<Posit> y;

		if(Layer<Posit>::training || !track_running_stats) {
			// calculate mean and variance
			StdTensor<Posit> mean(num_features);
			StdTensor<Posit> variance(num_features);
			calculate_mean_variance(x, mean, variance);

			if(track_running_stats) {
				// update running_mean_variance
				update_running_mean_variance(mean, variance);
			}
			
			// normalize
			y = normalize(x, mean, variance);	
			x_norm = y;
		}
		else {
			y = normalize(x, running_mean, running_variance);	
		}

		//scale and shift
		if(affine) {
			y *= gamma;
			y += beta;
		}

		return y;
	}

	StdTensor<Posit> backward(StdTensor<Posit> delta) {
		if(affine)
			gradient(delta);

		size_t const batch_size = delta.shape()[0];

		StdTensor<Posit> delta_1(x_norm.shape());
		delta *= gamma;
		StdTensor<Posit> temp1 = dot(delta, x_norm);
		StdTensor<Posit> temp2 = sum_first(delta);

		quire<Posit::nbits, Posit::es, Posit::nbits-1> q;
		for(size_t i=0, j=0, size=delta_1.size(); i<size; i++, j++) {
			if(j>=num_features)
				j=0;
			q = sw::unum::quire_mul(delta[i], Posit(batch_size));
			q -= sw::unum::quire_mul(x_norm[i], temp1[j]);
			q -= temp2[j];
			convert(q.to_value(), delta_1[i]);
		}

		delta_1 /= (stddev*batch_size);

		return delta_1;

		/*
		StdTensor<Posit> dx_norm = delta * gamma;
		delta = dx_norm * batch_size;
		delta -= sum_first(dx_norm);
		delta -= x_norm * dot(dx_norm, x_norm, 0);
		delta /= stddev * batch_size;
		return delta;
		*/
	}

	void gradient(StdTensor<Posit> delta) {
		StdTensor<Posit> temp_beta_gradient = sum_first(delta);

		//delta *= x_norm;
		//StdTensor<Posit> temp_gamma_gradient = sum_first(delta);
		StdTensor<Posit> temp_gamma_gradient = dot(delta, x_norm, 0);

		if(delta.dim()>1 && delta.shape()[0]>1) {
			temp_beta_gradient /= delta.shape()[0];
			temp_gamma_gradient /= delta.shape()[0];
		}

		beta_gradient += temp_beta_gradient;
		gamma_gradient += temp_gamma_gradient;

		return;
	}

	void calculate_mean_variance(	StdTensor<Posit>& x,
									StdTensor<Posit>& mean,
									StdTensor<Posit>& variance	) {

		constexpr size_t nbits = Posit::nbits;
		constexpr size_t es = Posit::es;

		size_t const size = x.size();
		size_t const batch_size = x.shape()[0];
		
		/*
		quire<nbits, es, Posit::nbits-1> sum, sq_sum;

		for(size_t i=0; i<num_features; i++){
			sum = sq_sum = 0;
			for(size_t j=i; j<size; j+=num_features){
				sum += x[j];
				sq_sum += quire_mul(x[j], x[j]);
			}
			convert(sum.to_value(), mean[i]);
			convert(sq_sum.to_value(), variance[i]);

			mean[i] /= batch_size;

			variance[i] /= batch_size;
			variance[i] -= mean[i]*mean[i];
		}
		*/

		quire<nbits, es, nbits-1> sum;

		// Calculate mean
		for(size_t i=0; i<num_features; i++){
			sum = 0;
			for(size_t j=i; j<size; j+=num_features){
				sum += x[j];
			}
			convert(sum.to_value(), mean[i]);

			mean[i] /= batch_size;
		}

		// Calculate variance
		for(size_t i=0; i<num_features; i++){
			sum = 0;
			for(size_t j=i; j<size; j+=num_features){
				Posit delta = x[j] - mean[i];
				sum += quire_mul(delta, delta);
			}
			convert(sum.to_value(), variance[i]);

			variance[i] /= batch_size;
		}

		return;
	}

	StdTensor<Posit> normalize(	StdTensor<Posit>& x,
								StdTensor<Posit>& mean,
								StdTensor<Posit>& variance	) {

		// Calculate standard deviation
		for(size_t i=0; i<num_features; i++) {
			stddev[i] = sqrt(variance[i]+eps);
		}

		size_t const size = x.size();
		StdTensor<Posit> y = x;

		for(size_t i=0, j=0; i<size; i++, j++) {
			if(j >= num_features)
				j = 0;
			// Subtract mean
			y[i] -= mean[j];
			// Divide by standard deviation
			y[i] /= stddev[j];
		}

		return y;
	}

	void update_running_mean_variance(	StdTensor<Posit>& mean,
										StdTensor<Posit>& variance	) {

		/*
		for(size_t i=0, size=mean.size(); i<size; i++) {
			running_mean[i] = (1-momentum)*running_mean[i] + momentum*mean[i];
			running_variance[i] = (1-momentum)*running_variance[i] + momentum*variance[i];
		}
		*/

		fused(running_mean, mean, 1-momentum, momentum);
		fused(running_variance, variance, 1-momentum, momentum);
	}

	template <typename PositFile=Posit>
	void write(std::ostream& out) {
		Layer<Posit>::write(out);	
		running_mean.template write<PositFile>(out);
		running_variance.template write<PositFile>(out);
	}

	template <typename PositFile=Posit>
	void read(std::istream& in) {
		Layer<Posit>::read(in);	
		running_mean.template read<PositFile>(in);
		running_variance.template read<PositFile>(in);
	}

private:
	size_t const num_features;
	Posit eps;
	Posit momentum;
	bool affine;
	bool track_running_stats;

	StdTensor<Posit> gamma;
	StdTensor<Posit> beta;
	StdTensor<Posit> gamma_gradient;
	StdTensor<Posit> beta_gradient;

	StdTensor<Posit> running_mean;
	StdTensor<Posit> running_variance;

	StdTensor<Posit> stddev;
	StdTensor<Posit> x_norm;
};

#endif /* BATCHNORM1D_HPP */

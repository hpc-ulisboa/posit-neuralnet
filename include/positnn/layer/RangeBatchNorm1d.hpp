#ifndef RANGEBATCHNORM1D_HPP
#define RANGEBATCHNORM1D_HPP

// General headers
#include <universal/posit/posit>

// Custom headers
#include "Layer.hpp"
#include "../tensor/matrix.hpp"
#include "../tensor/sum.hpp"
#include "../tensor/StdTensor.hpp"
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

template <typename Posit>
class RangeBatchNorm1d : public Layer<Posit> {
public:
	RangeBatchNorm1d(	size_t _num_features, Posit _eps=minpos<Posit::nbits, Posit::es>(),
						Posit _momentum=0.1, bool _affine=true, bool _track_running_stats=true	) :
		num_features(_num_features),
		eps(_eps), momentum(_momentum),
		affine(_affine), track_running_stats(_track_running_stats),
		gamma(_num_features), beta(_num_features),   
		gamma_gradient(_num_features), beta_gradient(_num_features),
		scale(_num_features),
		max_size(_num_features),
		min_size(_num_features)
	{
		if(track_running_stats){
			running_mean = StdTensor<Posit>(num_features);
			running_scale = StdTensor<Posit>(num_features);
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
		running_scale.set(Posit(1));
	}

	StdTensor<Posit> forward(StdTensor<Posit> x) {
		size_t const batch_size = x.shape()[0];

		C_1 = sqrt(2*log(Posit(batch_size)));

		if(Layer<Posit>::training || !track_running_stats) {
			// calculate mean and range
			StdTensor<Posit> mean = calculate_mean(x);
			x -= mean;

			// TODO: calculate range with x or x-mean?
			StdTensor<Posit> range = calculate_range(x);

			// calculate scale and normalize
			normalize(x, range);
			x_norm = x;

			if(track_running_stats) {
				// update running mean and scale
				update_running(mean);
			}
		}
		else {
			x -= running_mean;
			// calculate scale and normalize
			normalize(x, running_scale);
		}

		//scale and shift
		if(affine) {
			x *= gamma;
			x += beta;
		}

		return x;
	}

	StdTensor<Posit> backward(StdTensor<Posit> delta) {
		if(affine)
			gradient(delta);

		StdTensor<Posit> delta_1(x_norm.shape());
		size_t const size = delta_1.size();
		size_t const batch_size = delta.shape()[0];

		delta *= gamma;
		StdTensor<Posit> temp1 = sum_first(delta);
		StdTensor<Posit> temp2 = dot(delta, x_norm);
		temp2 /= C_1;

		Quire<Posit::nbits, Posit::es> q;
		for(size_t i=0, j=0; i<size; i++, j++) {
			if(j>=num_features)
				j=0;

			q = delta[i];
			q -= Quire_mul(temp1[j], 1/Posit(batch_size));
			if (max_min_idx[i] == 1) {
				q -= Quire_mul(temp2[j], 1/Posit(max_size[j]));
			}
			else if (max_min_idx[i] == -1) {
				q += Quire_mul(temp2[j], 1/Posit(min_size[j]));
			}
			convert(q.to_value(), delta_1[i]);
		}

		delta_1 /= scale;

		return delta_1;
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

	StdTensor<Posit> calculate_mean(StdTensor<Posit>& x) {
		constexpr size_t nbits = Posit::nbits;
		constexpr size_t es = Posit::es;

		size_t const size = x.size();
		size_t const batch_size = x.shape()[0];

		StdTensor<Posit> mean(num_features);

		Quire<nbits, es> sum;

		// Calculate mean
		for(size_t i=0; i<num_features; i++){
			sum = 0;
			for(size_t j=i; j<size; j+=num_features){
				sum += x[j];
			}
			convert(sum.to_value(), mean[i]);

			mean[i] /= batch_size;
		}

		return mean;
	}

	StdTensor<Posit> calculate_range(StdTensor<Posit>& x_mean) {
		size_t const size = x_mean.size();
		size_t const batch_size = x_mean.shape()[0];

		StdTensor<Posit> x_mean_T = transpose(x_mean);
		typename StdTensor<Posit>::iterator const begin = x_mean_T.begin();

		std::vector<Posit> max(num_features);
		std::vector<Posit> min(num_features);
		StdTensor<Posit> range(num_features);

		// Calculate range
		for(size_t i=0, j=0; i<num_features; i++, j+=batch_size){
			max[i] = *std::max_element(begin+j, begin+j+num_features);
			min[i] = *std::min_element(begin+j, begin+j+num_features);
			range[i] = max[i] - min[i];
		}

		max_min_idx = StdTensor<char>(x_mean.shape());

		// TODO: do i need to clear?
		//max_idx.set(false);
		//min_idx.set(false);

		std::fill(max_size.begin(), max_size.end(), 0);
		std::fill(min_size.begin(), min_size.end(), 0);

		for(size_t i=0, j=0; i<size; i+=batch_size, j++) {
			if(range[j].iszero()){
				continue;
			}

			for(size_t k=0; k<batch_size; k++) {
				if(x_mean_T[i+k] == max[j]) {
					max_min_idx[k*num_features+j] = 1;		
					max_size[j]++;
				}
				else if(x_mean_T[i+k] == min[j]) {
					max_min_idx[k*num_features+j] = -1;		
					min_size[j]++;
				}
			}
		}

		return range;
	}

	void normalize(StdTensor<Posit>& x, const StdTensor<Posit>& range) {

		for(size_t i=0; i<num_features; i++) {
			if(range[i].iszero())
				scale[i] = eps / C_1;
			else
				// TODO: multiply by C_1 or divide by C_1 (due to precision)
				scale[i] = range[i] / C_1;
		}

		x /= scale;
	}

	void update_running(StdTensor<Posit>& mean) {
		/*
		for(size_t i=0, size=mean.size(); i<size; i++) {
			running_mean[i] = (1-momentum)*running_mean[i] + momentum*mean[i];
			running_scale[i] = (1-momentum)*running_scale[i] + momentum*scale[i];
		}
		*/
		fused(running_mean, mean, 1-momentum, momentum);
		fused(running_scale, scale, 1-momentum, momentum);
	}

	template <typename PositFile=Posit>
void write(std::ostream& out) {
	Layer<Posit>::write(out);	
		running_mean.template write<PositFile>(out);
		running_scale.template write<PositFile>(out);
	}

	template <typename PositFile=Posit>
	void read(std::istream& in) {
		Layer<Posit>::read(in);	
		running_mean.template read<PositFile>(in);
		running_scale.template read<PositFile>(in);
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
	StdTensor<Posit> running_scale;

	StdTensor<Posit> scale;
	StdTensor<char> max_min_idx;
	std::vector<size_t> max_size;
	std::vector<size_t> min_size;
	StdTensor<Posit> x_norm;
	Posit C_1;
};

#endif /* RANGEBATCHNORM1D_HPP */

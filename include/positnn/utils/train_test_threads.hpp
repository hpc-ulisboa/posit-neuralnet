#ifndef TRAIN_TEST_THREADS_HPP
#define TRAIN_TEST_THREADS_HPP

#ifdef HL_THREADS
	#if HL_THREADS>1
		#define USING_HL_THREADS
	#endif
#endif /* HL_THREADS */

// General headers
#ifdef USING_HL_THREADS
#include <thread>
#endif /* USING_HL_THREADS */
#include <numeric>
#include <universal/posit/posit>

// Custom headers
#include "../tensor/StdTensor.hpp"

template <typename Loss, typename Model, typename Posit, typename targetT>
void batch_worker(	Model& master_model,
					StdTensor<Posit> const& batch_data,
					StdTensor<targetT> const& batch_target,
					size_t const batch_begin, size_t const batch_end,
					std::vector<StdTensor<Posit>>& gradients,
					float& loss_value	){

	// Local model
	Model model;
	model.train();
	
	// Copy master parameters
	copy_parameters(master_model.parameters(), model.parameters());

	// Slice batch
	StdTensor<Posit> data = batch_data.slice(batch_begin, batch_end);
	StdTensor<targetT> target = batch_target.slice(batch_begin, batch_end);

	// Forward pass
	auto output = model.forward(data);
	auto loss = Loss(output, target, Reduction::Sum);

	// Update loss of master
	loss_value = loss.template item<float>();

	// Backward pass and optimize
	model.zero_grad();
	loss.backward(model);

	// Copy gradients
	std::vector<Parameter<Posit>>& model_parameters = model.parameters();
	size_t nparameters = model_parameters.size();
	gradients.reserve(nparameters);

	for(size_t i=0; i<nparameters; i++){
		gradients.push_back(model_parameters[i].gradient);
	}
}

template <typename Loss, typename Model, typename Posit, typename targetT>
void forward_backward(	Model& model,
						StdTensor<Posit> data,
						StdTensor<targetT> target,
						std::vector<float>& losses,
						std::vector<std::vector<StdTensor<Posit>>>& gradients){
	
	// Batch size
	size_t const batch_size = target.shape()[0];

	// Batch threads
	size_t const batch_workers = (HL_THREADS<batch_size) ? HL_THREADS : batch_size;
	std::vector<std::thread> workers_threads;
	workers_threads.reserve(batch_workers);
	 
	// Related variables
	losses.resize(batch_workers);
	gradients.resize(batch_workers);
	
	// Distribute load (slice batch)
	size_t const worker_samples = batch_size / batch_workers;
	size_t const overloaded_workers = batch_size % batch_workers;

	size_t batch_begin, batch_end=0;

	// Initialize batch threads
	for(size_t t=0; t<batch_workers; t++){
		batch_begin = batch_end;

		size_t const nsamples = (t < overloaded_workers) ?
									worker_samples+1 : worker_samples;

		batch_end += nsamples;

		workers_threads.push_back(std::thread(batch_worker<Loss, Model, Posit, targetT>,
										std::ref(model),
										std::cref(data), std::cref(target),
										batch_begin, batch_end,
										std::ref(gradients[t]),
										std::ref(losses[t])));
	}

	for(std::thread& t : workers_threads)
		t.join();

	return;
}

template <size_t nbits, size_t es, size_t capacity=nbits-1>
void gradient_worker(	std::vector<Parameter<posit<nbits, es>>>& model_parameters,
						std::vector<std::vector<StdTensor<posit<nbits, es>>>>& gradients,
						size_t const elem_begin, size_t const nelem){
	
	size_t const ngradients = gradients.size();

	size_t n = elem_begin;
	size_t counter = 0;
	posit<nbits, es> aux;

	for(size_t i=0, size=model_parameters.size(); i<size && counter<nelem; i++){
		size_t parameter_size = model_parameters[i].gradient.size();

		while(n<parameter_size){
			quire<nbits, es, capacity> q;
			q.reset();

			for(std::vector<StdTensor<posit<nbits, es>>>& worker_gradients : gradients){
				q += worker_gradients[i][n];
			}

			convert(q.to_value(), aux);
			aux /= ngradients;

			model_parameters[i].gradient[n] = aux;
			
			n++;
			counter++;
		}

		n -= parameter_size;
	}
	
	return;
}

template <size_t nbits, size_t es, size_t capacity=nbits-1>
void sum_gradients(	std::vector<Parameter<posit<nbits, es>>>& model_parameters,
					std::vector<std::vector<StdTensor<posit<nbits, es>>>>& gradients	){

	// Total number of elements to sum of parameters
	size_t const nelem = std::accumulate(model_parameters.begin(), model_parameters.end(), 0,
		[](size_t sum, Parameter<posit<nbits, es>> const& parameter){ return sum + parameter.gradient.size(); });

	for(size_t i=0, size=model_parameters.size(); i<size; i++){
		for(auto worker_gradients : gradients){
			model_parameters[i].gradient += worker_gradients[i];
		}
	}
	
	// Gradient threads
	size_t const gradient_workers = (HL_THREADS<nelem) ? HL_THREADS : nelem;
	std::vector<std::thread> workers_threads;
	workers_threads.reserve(gradient_workers);
	 
	// Distribute load (distribute elements)
	size_t const worker_nelem = nelem / gradient_workers;
	size_t const overloaded_workers = nelem % gradient_workers;

	size_t elem_begin=0;

	// Initialize gradient threads
	for(size_t t=0; t<gradient_workers; t++){
		size_t const nelem = (t < overloaded_workers) ?
								worker_nelem+1 : worker_nelem;

		workers_threads.push_back(std::thread(gradient_worker<nbits, es, capacity>,
										std::ref(model_parameters), std::ref(gradients),
										elem_begin, nelem));

		elem_begin += nelem;
	}

	for(std::thread& t : workers_threads)
		t.join();

	return;
}

template <typename Loss, typename Model, typename Posit, typename targetT>
void forward_worker(	Model& master_model,
						StdTensor<Posit> const& batch_data,
						StdTensor<targetT> const& batch_target,
						StdTensor<targetT>& pred,
						size_t const batch_begin, size_t const batch_end,
						float& loss_value	){

	// Local model
	Model model;
	model.eval();
	
	// Copy master parameters
	copy_parameters(master_model.parameters(), model.parameters());

	// Slice batch
	StdTensor<Posit> data = batch_data.slice(batch_begin, batch_end);
	StdTensor<targetT> target = batch_target.slice(batch_begin, batch_end);

	// Forward pass
	auto output = model.forward(data);
	auto loss = Loss(output, target, Reduction::Sum);

	// Update loss of master
	loss_value = loss.template item<float>();

	// Get prediction from output
	auto local_pred = output.template argmax<targetT>(1);
	
	// Move to pred Tensor
	std::vector<targetT> from = local_pred.vector();
	std::vector<targetT> to = pred.vector();
	size_t offset = batch_begin * pred.strides()[0];
	std::move(from.begin(), from.end(), to.begin()+offset);
}

template <typename Loss, typename Model, typename Posit, typename targetT>
StdTensor<targetT> forward(	Model& model,
							StdTensor<Posit> data,
							StdTensor<targetT> target,
							std::vector<float>& losses	){
	
	// Batch size
	size_t const batch_size = target.shape()[0];

	// Batch threads
	size_t const batch_workers = (HL_THREADS<batch_size) ? HL_THREADS : batch_size;
	std::vector<std::thread> workers_threads;
	workers_threads.reserve(batch_workers);
	 
	// Related variables
	losses.resize(batch_workers);
	
	// Distribute load (slice batch)
	size_t const worker_samples = batch_size / batch_workers;
	size_t const overloaded_workers = batch_size % batch_workers;

	StdTensor<targetT> pred(target.shape());
	size_t batch_begin, batch_end=0;

	// Initialize batch threads
	for(size_t t=0; t<batch_workers; t++){
		batch_begin = batch_end;

		size_t const nsamples = (t < overloaded_workers) ?
									worker_samples+1 : worker_samples;

		batch_end += nsamples;

		workers_threads.push_back(std::thread(forward_worker<Loss, Model, Posit, targetT>,
										std::ref(model),
										std::cref(data), std::cref(target),
										std::ref(pred),
										batch_begin, batch_end,
										std::ref(losses[t])	));
	}

	for(std::thread& t : workers_threads)
		t.join();

	return pred;
}




#endif /* TRAIN_TEST_THREADS_HPP */

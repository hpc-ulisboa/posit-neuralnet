#ifndef MATRIX_HPP
#define MATRIX_HPP

#ifdef LL_THREADS
	#if LL_THREADS>1
		#define USING_LL_THREADS
	#endif
#endif /* LL_THREADS */

// General headers
#ifdef USING_LL_THREADS
#include <thread>
#endif /* USING_LL_THREADS */
#include <universal/posit/posit>
#include <vector>

// Custom headers
#include "StdTensor.hpp"
#include "../utils/Quire.hpp"

// Namespaces
using namespace sw::unum;

// Matrix transpose
template <typename T>
StdTensor<T> transpose(const StdTensor<T>& a, const size_t block=4){
	// TODO: throw error if dim>2
	const size_t row = a.shape()[0];
	const size_t col = a.shape()[1];

	StdTensor<T> c({col, row});

    for (size_t bi=0; bi<row; bi+=block) {
		size_t imax = (bi+block>row ? row : bi+block);
        for(size_t bj=0; bj<col; bj+=block) {
			size_t jmax = (bj+block>col ? col : bj+block);
			for(size_t i=bi; i<imax; i++) {
				for(size_t j=bj; j<jmax; j++) {
					c[j*row + i] = a[i*col + j];
				}
			}
		}
	}

	return c;
}

// Function that implements fused product (by constant) and add
template <size_t nbits, size_t es, size_t capacity=nbits-1>
void fused(	StdTensor<posit<nbits, es>>& a,
			const StdTensor<posit<nbits, es>>& b,
			const posit<nbits, es> alpha,
			const posit<nbits, es> beta	){
	// TODO: throw error if size(a) != size(b)

	Quire<nbits, es> q;
	bool const alpha1 = alpha.isone();
	bool const beta1 = beta.isone();

	for(size_t i=0, size=a.size(); i<size; i++) {
		if(alpha1)
			q = a[i];
		else
			q = Quire_mul(a[i], alpha);

		if(beta1)
			q += b[i];
		else
			q += Quire_mul(b[i], beta);

		convert(q.to_value(), a[i]);
	}
}

// Function that implements fused product (by constant) and add
// c = a * alpha + b
template <size_t nbits, size_t es, size_t capacity=nbits-1>
void fused(	const StdTensor<posit<nbits, es>>& a,
			const StdTensor<posit<nbits, es>>& b,
			StdTensor<posit<nbits, es>>& c,
			const posit<nbits, es> alpha) {
	// TODO: throw error if size(a) != size(b)

	Quire<nbits, es> q;
	if(alpha.isone()) {
		c = a + b;
	}
	else {
		value<1 + 2 * (nbits - es)> result;
		for(size_t i=0, size=a.size(); i<size; i++) {
			result = fma(a[i], alpha, b[i]);
			convert(result, c[i]);
		}
	}
}

#ifdef USING_LL_THREADS

// Function to be executed by each thread to sum along axis
template <size_t nbits, size_t es, size_t capacity=nbits-1>
void dot_thread (	const StdTensor<posit<nbits, es>>& a,
					const StdTensor<posit<nbits, es>>& b,
					StdTensor<posit<nbits, es>>& c,
					const size_t i_begin, size_t i_end,
					const size_t j_first, size_t j_last,
					const size_t c_begin,
					const size_t loop_stride, const size_t stride, const size_t axis_size){

	if(j_last == 0)			// to protect when j_end=0, which occurs when noffset=0
		j_last = stride;
	else					// to protect when i_begin==i_end, which occurs when nblocks=0
		i_end += loop_stride;

	Quire<nbits, es> q;
	size_t n = c_begin;

	for(size_t i=i_begin; i<i_end; i+=loop_stride) {	// loop blocks
		const size_t j_begin = (i==i_begin) ? j_first : 0;
		const size_t j_end = (i==i_end-loop_stride) ? j_last : stride;

		for(size_t j=j_begin; j<j_end; j++){	// loop beginning elements of block
			q.clear();

			for(size_t k=i+j, l=0; l<axis_size; k+=stride, l++){	// loop elements to sum
				q += Quire_mul(a[k], b[k]);
			}

			convert(q.to_value(), c[n++]);
		}
	}

	return;
}

// Matrix sum along axis using threads
template <size_t nbits, size_t es, size_t capacity=nbits-1>
StdTensor<posit<nbits, es>> dot(const StdTensor<posit<nbits, es>>& a,
								const StdTensor<posit<nbits, es>>& b,
								const size_t axis=0){
	std::vector<size_t> new_shape;

	if (a.dim()==1) {
		new_shape.push_back(1);
	}
	else {
		new_shape = a.shape();
		new_shape.erase(new_shape.begin()+axis);
	}

	// TODO: THROW ERROR IF MATRIX DIMENSIONS ARE INVALID
	StdTensor<posit<nbits, es>> c(new_shape);
	const size_t size = c.size();

	const size_t max_threads = (LL_THREADS<size) ? LL_THREADS : size;
	std::vector<std::thread> threads;
	threads.reserve(max_threads);

	size_t const axis_size = a.shape()[axis];
	size_t const stride = a.strides()[axis];
	size_t const loop_stride = axis_size*stride;

	size_t const nelem = size / max_threads;
	size_t const nthreads_more = size % max_threads;
	size_t const nblocks = nelem / stride;
	size_t const noffset = nelem % stride;

	size_t i_begin, j_begin, c_begin=0;
	size_t i_end=0, j_end=0;

	for(size_t t=0; t<max_threads; t++){
		i_begin = i_end;
		j_begin = j_end;

		if(t < nthreads_more)
			j_end += noffset+1;
		else
			j_end += noffset;


		if(j_end >= stride) {
			i_end += (nblocks+1)*loop_stride;
			j_end -= stride;
		}
		else {
			i_end += nblocks*loop_stride;
		}
		
		threads.push_back(std::thread(dot_thread<nbits, es, capacity>,
										std::cref(a), std::cref(b), std::ref(c),
										i_begin, i_end,
										j_begin, j_end,
										c_begin,
										loop_stride, stride, axis_size));

		if(t < nthreads_more)
			c_begin += nelem+1;
		else
			c_begin += nelem;

		//getchar();
	}

	//getchar();
	
	for(std::thread& t : threads) {
		t.join();
	}

	return c;
}

#else

// Matrix sum along axis
template <size_t nbits, size_t es, size_t capacity=nbits-1>
StdTensor<posit<nbits, es>> dot(const StdTensor<posit<nbits, es>>& a,
								const StdTensor<posit<nbits, es>>& b,
								const size_t axis=0){

	std::vector<size_t> new_shape = a.shape();
	new_shape.erase(new_shape.begin()+axis);
	if (new_shape.empty())
		new_shape.push_back(1);

	StdTensor<posit<nbits, es>> c(new_shape);

	Quire<nbits, es> q;
	
	size_t i, j, k, l, n=0;
	size_t const size = a.size();
	size_t const axis_size = a.shape()[axis];
	size_t const stride = a.strides()[axis];
	size_t const loop_stride = axis_size*stride;

	for(i=0; i<size; i+=loop_stride) {	// loop blocks
		for(j=0; j<stride; j++){	// loop beginning elements of block
			for(k=i+j, l=0, q=0; l<axis_size; k+=stride, l++){	// loop elements to sum
				q += Quire_mul(a[k], b[k]);
			}
			convert(q.to_value(), c[n++]);
		}
	}

	return c;
}

#endif /* USING_LL_THREADS */

#ifdef USING_LL_THREADS

// Function to be executed by each thread to multiply rows
template <size_t nbits, size_t es, size_t capacity=nbits-1>
void matmul_row_thread (const StdTensor<posit<nbits, es>>& a,
						const StdTensor<posit<nbits, es>>& b,
						StdTensor<posit<nbits, es>>& c,
						const size_t a_begin, size_t a_end,
						const size_t b_begin, size_t b_end, const size_t b_size,
						const size_t c_begin, const size_t stride){

	if(b_end == 0)			// to protect when b_end=0, which occurs when ncols=0
		b_end = b_size;
	else					// to protect when a_begin==a_end, which occurs when nrows=0
		a_end += stride;

	Quire<nbits, es> q;
	size_t n = c_begin;

	//std::cout << "a_begin=" << a_begin << " a_end=" << a_end << " b_begin=" << b_begin << " b_end=" << b_end << " c_begin=" << c_begin << std::endl;

	for(size_t i=a_begin; i<a_end; i+=stride){
		const size_t j_begin = (i==a_begin) ? b_begin : 0;
		const size_t j_end = (i==a_end-stride) ? b_end : b_size;

		for(size_t j=j_begin; j<j_end; j+=stride){
			q.clear();

			for(size_t k=0; k<stride; k++){
				// TODO: try changing indices to relative instead of absolute
				q += Quire_mul(a[i+k], b[j+k]);
			}

			convert(q.to_value(), c[n++]);
		}
	}

	return;
}

// Matrix multiplication of rows using threads. Equivalent to A * B^T
template <size_t nbits, size_t es, size_t capacity=nbits-1>
StdTensor<posit<nbits, es>> matmul_row (const StdTensor<posit<nbits, es>>& a, const StdTensor<posit<nbits, es>>& b){
	// TODO: THROW ERROR IF MATRIX DIMENSIONS ARE INVALID
	const size_t rows = a.shape()[0];
	const size_t cols = b.shape()[0];
	const size_t stride = a.strides()[0];
	const size_t b_size = b.size();

	StdTensor<posit<nbits, es>> c({rows, cols});
	const size_t size = c.size();

	const size_t max_threads = (LL_THREADS<size) ? LL_THREADS : size;
	std::vector<std::thread> threads;
	threads.reserve(max_threads);

	const size_t nelem = size / max_threads;
	const size_t nthreads_more = size % max_threads;
	const size_t nrows = nelem / cols;
	const size_t ncols = nelem % cols;

	size_t a_begin, b_begin;
	size_t a_end=0, b_end=0;

	//std::cout << "rows=" << rows << " cols=" << cols << " stride=" << stride << " b_size=" << b_size << " size=" << size << std::endl;

	for(size_t t=0; t<max_threads; t++){
		a_begin = a_end;
		b_begin = b_end;

		a_end += nrows;

		if(t < nthreads_more)
			b_end += ncols+1;
		else
			b_end += ncols;

		if(b_end >= cols) {
			a_end++;
			b_end -= cols;
		}

		threads.push_back(std::thread(matmul_row_thread<nbits, es, capacity>,
										std::cref(a), std::cref(b), std::ref(c),
										a_begin*stride, a_end*stride,
										b_begin*stride, b_end*stride, b_size,
										a_begin*cols+b_begin, stride));

		//getchar();
	}
	
	for(std::thread& t : threads) {
		t.join();
	}

	return c;
}

#else

// Matrix multiplication of rows. Equivalent to A * B^T
template <size_t nbits, size_t es, size_t capacity=nbits-1>
StdTensor<posit<nbits, es>> matmul_row(const StdTensor<posit<nbits, es>>& a, const StdTensor<posit<nbits, es>>& b){
	// TODO: THROW ERROR IF MATRIX DIMENSIONS ARE INVALID
	StdTensor<posit<nbits, es>> c({a.shape()[0], b.shape()[0]});

	Quire<nbits, es> q;

	const size_t a_size = a.size();
	const size_t b_size = b.size();
	const size_t stride = a.strides()[0];

	size_t i, j, k, n=0;

	for(i=0; i<a_size; i+=stride){
		for(j=0; j<b_size; j+=stride){
			for(q=0, k=0; k<stride; k++){
				// TODO: try changing indices to relative instead of absolute
				q += Quire_mul(a[i+k], b[j+k]);
			}
			convert(q.to_value(), c[n++]);
		}
	}

	return c;
}

#endif /* USING_LL_THREADS */

#ifdef USING_LL_THREADS

// Function to be executed by each thread to multiply rows
template <size_t nbits, size_t es, size_t capacity=nbits-1>
void matmul_row_add_thread (const StdTensor<posit<nbits, es>>& a,
							const StdTensor<posit<nbits, es>>& b,
							const StdTensor<posit<nbits, es>>& c,
							StdTensor<posit<nbits, es>>& d,
							const size_t a_begin, size_t a_end,
							const size_t b_begin, size_t b_end, const size_t b_size,
							const size_t c_begin, const size_t c_size,
							const size_t d_begin, const size_t stride){

	if(b_end == 0)			// to protect when b_end=0, which occurs when ncols=0
		b_end = b_size;
	else					// to protect when a_begin==a_end, which occurs when nrows=0
		a_end += stride;

	Quire<nbits, es> q;
	size_t l = c_begin;
	size_t n = d_begin;

	for(size_t i=a_begin; i<a_end; i+=stride){
		const size_t j_begin = (i==a_begin) ? b_begin : 0;
		const size_t j_end = (i==a_end-stride) ? b_end : b_size;

		if(l >= c_size)
			l = 0;

		for(size_t j=j_begin; j<j_end; j+=stride){
			q.clear();

			for(size_t k=0; k<stride; k++){
				// TODO: try changing indices to relative instead of absolute
				q += Quire_mul(a[i+k], b[j+k]);
			}

			//std::cerr << "c_index=" << l << "/" << c.size() << " d_index=" << n << "/" << c.size() << std::endl;
			q += c[l++];
			convert(q.to_value(), d[n++]);
		}
	}

	return;
}

// Matrix multiplication of rows and addition. Equivalent to D = A * B^T + C
template <size_t nbits, size_t es, size_t capacity=nbits-1>
StdTensor<posit<nbits, es>> matmul_row_add(const StdTensor<posit<nbits, es>>& a, const StdTensor<posit<nbits, es>>& b, const StdTensor<posit<nbits, es>>& c){
	// TODO: THROW ERROR IF MATRIX DIMENSIONS ARE INVALID
	const size_t rows = a.shape()[0];
	const size_t cols = b.shape()[0];
	const size_t stride = a.strides()[0];
	const size_t b_size = b.size();
	const size_t c_size = c.size();

	StdTensor<posit<nbits, es>> d({rows, cols});
	const size_t size = d.size();

	const size_t max_threads = (LL_THREADS<size) ? LL_THREADS : size;
	std::vector<std::thread> threads;
	threads.reserve(max_threads);

	const size_t nelem = size / max_threads;
	const size_t nthreads_more = size % max_threads;
	const size_t nrows = nelem / cols;
	const size_t ncols = nelem % cols;

	size_t a_begin, b_begin, c_begin, d_begin;
	size_t a_end=0, b_end=0;

	//std::cerr << "rows=" << rows << " cols=" << cols << " stride=" << stride << " b_size=" << b_size << " size=" << size << std::endl;

	for(size_t t=0; t<max_threads; t++){
		a_begin = a_end;
		b_begin = b_end;

		a_end += nrows;

		if(t < nthreads_more)
			b_end += ncols+1;
		else
			b_end += ncols;

		if(b_end >= cols) {
			a_end++;
			b_end -= cols;
		}

		d_begin = a_begin * cols + b_begin;

		if(d_begin < c_size)
			c_begin = d_begin;
		else
			c_begin = b_begin;

		//std::cerr << "a_begin=" << a_begin << " a_end=" << a_end << " b_begin=" << b_begin << " b_end=" << b_end << " c_begin=" << c_begin << " c_size=" << c_size << " d_begin=" << d_begin << " stride=" << stride << std::endl;
	
		threads.push_back(std::thread(matmul_row_add_thread<nbits, es, capacity>,
										std::cref(a), std::cref(b), std::cref(c), std::ref(d),
										a_begin*stride, a_end*stride,
										b_begin*stride, b_end*stride, b_size,
										c_begin, c_size,
										d_begin, stride));

		//getchar();
	}
	
	for(std::thread& t : threads) {
		t.join();
	}

	return d;
}

#else

// Matrix multiplication of rows and addition. Equivalent to D = A * B^T + C
template <size_t nbits, size_t es, size_t capacity=nbits-1>
StdTensor<posit<nbits, es>> matmul_row_add(const StdTensor<posit<nbits, es>>& a, const StdTensor<posit<nbits, es>>& b, const StdTensor<posit<nbits, es>>& c){
	// TODO: THROW ERROR IF MATRIX DIMENSIONS ARE INVALID
	StdTensor<posit<nbits, es>> d({a.shape()[0], b.shape()[0]});

	Quire<nbits, es> q;	// TODO: COMPUTE BEST CAPACITY

	const size_t a_size = a.size();
	const size_t b_size = b.size();
	const size_t stride = a.strides()[0];
	const size_t c_size = c.size();

	size_t i, j, k, n=0;

	for(i=0; i<a_size; i+=stride){
		for(j=0; j<b_size; j+=stride){
			for(q=0, k=0; k<stride; k++){
				// TODO: try changing indices to relative instead of absolute
				q += Quire_mul(a[i+k], b[j+k]);
			}
			q += c[n%c_size];
			convert(q.to_value(), d[n++]);
		}
	}

	return d;
}

#endif /* USING_LL_THREADS */

// Inline functions
// Matrix multiplication
template <size_t nbits, size_t es, size_t capacity=nbits-1>
inline StdTensor<posit<nbits, es>> matmul (const StdTensor<posit<nbits, es>>& a, const StdTensor<posit<nbits, es>>& b) {
	StdTensor<posit<nbits, es>> bT = transpose(b);
	return matmul_row<nbits, es, capacity>(a, bT);
}

// Matrix multiplication of columns. Equivalent to A^T * B
template <size_t nbits, size_t es, size_t capacity=nbits-1>
inline StdTensor<posit<nbits, es>> matmul_col (const StdTensor<posit<nbits, es>>& a, const StdTensor<posit<nbits, es>>& b) {
	StdTensor<posit<nbits, es>> aT = transpose(a);
	StdTensor<posit<nbits, es>> bT = transpose(b);
	return matmul_row<nbits, es, capacity>(aT, bT);
}

// Matrix multiplication and addition. Equivalent to D = A * B + C
template <size_t nbits, size_t es, size_t capacity=nbits-1>
inline StdTensor<posit<nbits, es>> matmul_add (const StdTensor<posit<nbits, es>>& a, const StdTensor<posit<nbits, es>>& b, const StdTensor<posit<nbits, es>>& c) {
	StdTensor<posit<nbits, es>> bT = transpose(b);
	return matmul_row_add<nbits, es, capacity>(a, bT, c);
}

// Matrix multiplication of columns and addition. Equivalent to D = A^T * B + C
template <size_t nbits, size_t es, size_t capacity=nbits-1>
inline StdTensor<posit<nbits, es>> matmul_col_add (const StdTensor<posit<nbits, es>>& a, const StdTensor<posit<nbits, es>>& b, const StdTensor<posit<nbits, es>>& c) {
	StdTensor<posit<nbits, es>> aT = transpose(a);
	StdTensor<posit<nbits, es>> bT = transpose(b);
	return matmul_row_add<nbits, es, capacity>(aT, bT, c);
}

/* OLD

// Matrix transpose - simple algorithm without multiplications for indices
template <typename T>
StdTensor<T> transpose(const StdTensor<T>& a){
	// TODO: throw error if dim>2
	const size_t row = a.shape()[0];
	const size_t col = a.shape()[1];
	const size_t size = a.size();

	StdTensor<T> c({col, row});

	size_t i, j, k;

    for (i=0, j=0, k=0; i<size; i++, j += row) {
		if(j>=size)
			j = ++k;

		c[j] = a[i];
	}

	return c;
}

// Matrix transpose - simple algorithm
template <typename T>
StdTensor<T> transpose(const StdTensor<T>& a){
	// TODO: throw error if dim>2
	const size_t row = a.shape()[0];
	const size_t col = a.shape()[1];

	StdTensor<T> c({col, row});

	size_t i, j;
		
    for (i=0; i<row; i++) {
        for(j=0; j<col; j++) {
			c[j*row + i] = a[i*col + j];
		}
	}

	return c;
}

// Matrix multiplication of rows using loop tiling/blocks
template <typename T, size_t capacity=10>
StdTensor<T> matmul_row(const StdTensor<T>& a, const StdTensor<T>& b, const size_t block=4){
	constexpr size_t nbits = T::nbits;
	constexpr size_t es = T::es;
	
	// TODO: THROW ERROR IF MATRIX DIMENSIONS ARE INVALID
	StdTensor<T> c({a.shape()[0], b.shape()[0]});
	const size_t size = c.size();

	using Quire = Quire<nbits, es>;
	std::vector<Quire> q(size);	// TODO: COMPUTE BEST CAPACITY

	const size_t a_lines = a.shape()[0];
	const size_t b_lines = b.shape()[0];
	const size_t line_size = a.shape()[1];

	for(size_t bi=0; bi<a_lines; bi+=block){
		size_t imax = (bi+block>a_lines ? a_lines : bi+block);
		for(size_t bj=0; bj<b_lines; bj+=block){
			size_t jmax = (bj+block>b_lines ? b_lines : bj+block);
			for(size_t bk=0; bk<line_size; bk+=block){
				size_t kmax = (bk+block>line_size ? line_size : bk+block);
				for(size_t i=bi; i<imax; i++) {
					for(size_t j=bj; j<jmax; j++) {
						for(size_t k=bk; k<kmax; k++) {
							q[i*b_lines+j] += Quire_mul(a[line_size*i+k], b[line_size*j+k]);
						}
					}
				}
			}
		}
	}

	for(size_t i=0; i<size; i++)
		convert(q[i].to_value(), c[i]);

	return c;
}

// Matrix multiplication - simple algorithm
template <typename T, size_t capacity=10>
StdTensor<T> matmul(const StdTensor<T>& a, const StdTensor<T>& b){
// TODO: CHECK IF TRANSPOSING B AND MULTIPLYING LINES(A,B) IS FASTER THAN SIMPLE MATRIX MULTIPLICATION
	constexpr size_t nbits = T::nbits;
	constexpr size_t es = T::es;
	
	// TODO: THROW ERROR IF MATRIX DIMENSIONS ARE INVALID
	StdTensor<T> c({a.shape()[0], b.shape()[1]});

	Quire<nbits, es> q;	// TODO: COMPUTE BEST CAPACITY

	size_t ix, iy;
	size_t i, j, k=0;
	
	const size_t a_size = a.size();
	const size_t a_stride = a.strides()[0];
	const size_t b_stride = b.strides()[0];

	for(i=0; i<a_size; i+=a_stride){
		for(j=0; j<b_stride; j++){
			for(q=0, ix=i, iy=j; ix<i+a_stride; ix++, iy+=b_stride){
				q += Quire_mul(a[ix], b[iy]);
			}
			convert(q.to_value(), c[k++]);
		}
	}

	return c;
}

// Matrix multiplication with loop tiling/blocks
template <typename T, size_t capacity=10>
StdTensor<T> matmul (const StdTensor<T>& a, const StdTensor<T>& b, const size_t block=4){
	constexpr size_t nbits = T::nbits;
	constexpr size_t es = T::es;

	const size_t a_rows = a.shape()[0];
	const size_t b_cols = b.shape()[1];
	const size_t size = a.shape()[1];
	
	// TODO: THROW ERROR IF MATRIX DIMENSIONS ARE INVALID
	StdTensor<T> c({a_rows, b_cols});
	const size_t nelem = c.size();

	using Quire = Quire<nbits, es>;
	std::vector<Quire> q(nelem);	// TODO: COMPUTE BEST CAPACITY

	for(size_t bj=0; bj<b_cols; bj+=block){
		size_t jmax = (bj+block>b_cols ? b_cols : bj+block);
		for(size_t bk=0; bk<size; bk+=block){
			size_t kmax = (bk+block>size ? size : bk+block);
			for(size_t bi=0; bi<a_rows; bi+=block){
				size_t imax = (bi+block>a_rows ? a_rows : bi+block);
				for(size_t j=bj; j<jmax; j++) {
					for(size_t i=bi; i<imax; i++) {
						for(size_t k=bk; k<kmax; k++) {
							q[i*b_cols+j] += Quire_mul(a[i*size+k], b[k*b_cols+j]);
						}
					}
				}
			}
		}
	}

	for(size_t i=0; i<nelem; i++)
		convert(q[i].to_value(), c[i]);

	return c;
}

// Matrix multiplication using transpose and multiplication of rows with loop tiling/blocks
template <typename T, size_t capacity=10>
inline StdTensor<T> matmul (const StdTensor<T>& a, const StdTensor<T>& b, size_t block=4) {
	StdTensor<T> bT = transpose(b);
	return matmul_row2<T, capacity>(a, bT, block);
}

// Matrix multiplication of columns - simple algorithm
template <typename T, size_t capacity=10>
StdTensor<T> matmul_col(const StdTensor<T>& a, const StdTensor<T>& b){
// Equivalent to A^T * B
	constexpr size_t nbits = T::nbits;
	constexpr size_t es = T::es;
	
	// TODO: THROW ERROR IF MATRIX DIMENSIONS ARE INVALID
	StdTensor<T> c({a.shape()[1], b.shape()[1]});

	Quire<nbits, es> q;	// TODO: COMPUTE BEST CAPACITY
	T sum;

	size_t i, j, k, n=0;

	const size_t a_stride = a.strides()[0];
	const size_t b_stride = b.strides()[0];
	const size_t size = a.shape()[0];

	// TODO: detect if posit. check if it throws error in Quire_mul with floats
	
	for(i=0; i<a_stride; i++){
		for(j=0; j<b_stride; j++){
			for(q=0, k=0; k<size; k++){
				q += Quire_mul(a[i+k*a_stride], b[j+k*b_stride]);
				// TODO: PROBLEM OF SLOW IS HERE ^
			}
			convert(q.to_value(), c[n++]);
		}
	}

	return c;
}

template <typename T>
StdTensor<T> matrix_add(const StdTensor<T>& a, const StdTensor<T>& b){
	StdTensor<T> c(a.shape());

	size_t i, a_size, b_size;
	a_size = a.size();
	b_size = b.size();

	for(i=0; i<a_size; i++){
		c[i] = a[i] + b[i%b_size];			// TODO: WARN THAT B IS REPEATED
	}

	return c;
}

template <typename T>
StdTensor<T> matrix_subtract(const StdTensor<T>& a, const StdTensor<T>& b){
	StdTensor<T> c(a.shape());

	size_t i, a_size, b_size;
	a_size = a.size();
	b_size = b.size();

	for(i=0; i<a_size; i++){
		c[i] = a[i] - b[i%b_size];			// TODO: WARN THAT B IS REPEATED
	}

	return c;
}

template <typename T>
StdTensor<T> matrix_hadamard(const StdTensor<T>& a, const StdTensor<T>& b){
	StdTensor<T> c(a.shape());

	size_t i, a_size, b_size;
	a_size = a.size();
	b_size = b.size();

	for(i=0; i<a_size; i++){
		c[i] = a[i] * b[i%b_size];			// TODO: WARN THAT B IS REPEATED
	}

	return c;
}

template <typename T, typename Scalar>
StdTensor<T> matrix_multiply_scalar(const StdTensor<T>& a, const Scalar& b){
	StdTensor<T> c(a.shape());

	size_t i, a_size;
	a_size = a.size();

	for(i=0; i<a_size; i++){
		c[i] = a[i] * b;
	}

	return c;
}

template <typename T, typename Scalar>
StdTensor<T> matrix_divide_scalar(const StdTensor<T>& a, const Scalar& b){
	StdTensor<T> c(a.shape());

	size_t i, a_size;
	a_size = a.size();

	for(i=0; i<a_size; i++){
		c[i] = a[i] / b;
	}

	return c;
}

*/

#endif /* MATRIX */

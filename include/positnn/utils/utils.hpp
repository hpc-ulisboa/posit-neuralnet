#ifndef UTILS_HPP
#define UTILS_HPP

// General headers
#include <cstdint>
#include <iostream>
#include <universal/posit/posit>
#include <vector>

// Namespaces
using namespace sw::unum;

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
	if(v.size()!=0){
		std::copy(v.begin(), v.end()-1, std::ostream_iterator<T>(out, " "));
		out << v.back();
	}
	return out;
}

template <typename T>
void write_vector(std::ostream& out, const std::vector<T>& vec) {
	typename std::vector<T>::size_type size = vec.size();
	out.write((char*)&size, sizeof(size));
	out.write((char*)vec.data(), size*sizeof(T));
}

template <typename T>
void read_vector(std::istream& in, std::vector<T>& vec) {
	typename std::vector<T>::size_type size;
	in.read((char*)&size, sizeof(size));
	vec.resize(size);
	in.read((char*)vec.data(), size*sizeof(T));
}

/*
template <size_t nbits>
void write_bitset(std::ostream out, std::bitset<nbits> b){
	if(nbits<=8) {
		uint8_t bytes1 = (uint8_t) b.to_ulong();
		out.write((char*)&bytes1, sizeof(bytes1));
	}
	else if(nbits<=16) {
		uint16_t bytes2 = (uint16_t) b.to_ulong();
		out.write((char*)&bytes2, sizeof(bytes2));
	}
	else if(nbits<=32) {
		uint32_t bytes4 = (uint32_t) b.to_ulong();
		out.write((char*)&bytes4, sizeof(bytes4));
	}
	else if(nbits<=64) {
		uint64_t bytes8 = (uint64_t) b.to_ullong();
		out.write((char*)&bytes8, sizeof(bytes8));
	}
	else
		throw std::invalid_argument( "Bitset has more than 64 bits. Cannot convert to integer." );
}
*/

template <typename Posit, typename PositFile>
void write_posit(std::ostream& out, const Posit& p){
	using byte = unsigned char;
	constexpr size_t nbits = PositFile::nbits;
	
	PositFile aux = p;
	bitblock<nbits> all_bytes = aux.get();

	for(size_t n=0; n<nbits; n+=8){
		byte one_byte {0};

		for(uint8_t i=0; i<8; i++){
			one_byte <<= 1;
			if(n+i<nbits && all_bytes[n+i])
				one_byte |= 0b1; 
		}

		out.write((char *)&one_byte, 1);
	}
}

template <typename Posit, typename PositFile>
void read_posit(std::istream& in, Posit& p){
	using byte = unsigned char;
	constexpr size_t nbits = PositFile::nbits;

	bitblock<nbits> all_bytes;
	PositFile aux;

	for(size_t n=0; n<nbits; n+=8){
		byte one_byte;
		in.read((char *)&one_byte, 1);

		byte mask = 0b10000000;

		for(uint8_t i=0; i<8; i++){
			if(one_byte & mask)
				all_bytes.set(n+i); 
			mask>>=1;
		}
	}

	aux.set(all_bytes);
	p = aux;
}

// TODO: save posit nbits and es

template <typename Posit, typename PositFile>
void write_vector_posit(std::ostream& out, const std::vector<Posit>& vec) {
	size_t const size = vec.size();
	out.write((char*)&size, sizeof(size));

	for(Posit const& elem : vec) {
		write_posit<Posit, PositFile>(out, elem);
	}
}

template <typename Posit, typename PositFile>
void read_vector_posit(std::istream& in, std::vector<Posit>& vec) {
	size_t size;
	in.read((char*)&size, sizeof(size));
	vec.resize(size);

	for(size_t i=0; i<size; i++){
		read_posit<Posit, PositFile>(in, vec[i]);
	}
}

template <size_t nbits, size_t es>
inline posit<nbits, es> sigmoid_approx(posit<nbits, es> p) {
	bitblock<nbits> bits = p.get();
	bits.flip(nbits-1);
	bits>>=2;
	p.set(bits);
	return p;
}

template <size_t nbits, size_t es>
inline posit<nbits, es> exp_approx(posit<nbits, es> p) {
// Only good between -2 and 2
	p = -p;
	sigmoid_approx(p);
	p = 1/p - 1;
	return p;
}

template <size_t nbits, size_t es>
inline posit<nbits, es> tanh_approx(posit<nbits, es> p) {
	return 2*sigmoid_approx(2*p)-1;
}

template<size_t nbits, size_t es>
posit<nbits, es> round_pow2(const posit<nbits, es>& number) {
	// If it's already a power of 2
	if(number.ispowerof2())
		return number;

	constexpr size_t fbits = (es + 2 >= nbits ? 0 : nbits - 3 - es);             // maximum number of fraction bits: derived
	bool s;
	scale(number);
	regime<nbits, es> r;
	exponent<nbits, es> e;
	fraction<fbits> f;
	bitblock<nbits> raw = number.get();
	extract_fields(raw, s, r, e, f);

	bitblock<fbits> fraction = f.get();
	size_t nrFracBits = f.nrBits();
	bool greaterHalf = s ^ fraction[fbits-1];

	// Round down
	if(!greaterHalf) {
		for(size_t i=0; i<nrFracBits; i++)
			raw.reset(i);
	}
	// Round up
	else {
		for(size_t i=0; i<nrFracBits; i++)
			raw.set(i);
		increment_bitset(raw);
	}
	
	posit<nbits, es> result;
	result.set(raw);

	//std::cout << "before:\t" << to_binary(number) << std::endl;
	//std::cout << "after:\t" << to_binary(result) << std::endl;

	return result;
}

// FAM (corrected): fused add-multiply: (a + b) * c
template<size_t nbits, size_t es>
value<2 * (nbits + 3 - es)> fam_corrected(const posit<nbits, es>& a, const posit<nbits, es>& b, const posit<nbits, es>& c) {
	constexpr size_t fbits = nbits - 3 - es;
	constexpr size_t abits = fbits + 4;       // size of the addend
	//constexpr size_t fhbits = fbits + 1;      // size of fraction + hidden bit
	constexpr size_t fhbits = abits + 1;      // size of fraction + hidden bit
	constexpr size_t mbits = 2 * (fhbits + 1);      // size of the multiplier output

	value<fbits> va, vb, ctmp;
	value<abits+1> sum, vc;
	value<mbits> product;

	// special case
	if (c.iszero()) return product;

	// first the add
	if (!a.iszero() || !b.iszero()) {
		// transform the inputs into (sign,scale,fraction) triples
		va.set(sign(a), scale(a), extract_fraction<nbits, es, fbits>(a), a.iszero(), a.isnar());;
		vb.set(sign(b), scale(b), extract_fraction<nbits, es, fbits>(b), b.iszero(), b.isnar());;

		module_add<fbits, abits>(va, vb, sum);    // multiply the two inputs
		if (sum.iszero()) return product;  // product is still zero
	}
	// second, the multiply		
	ctmp.set(sign(c), scale(c), extract_fraction<nbits, es, fbits>(c), c.iszero(), c.isnar());
	vc.template right_extend<fbits, abits+1>(ctmp); // right-extend the c argument and assign to multiplier input
	module_multiply(sum, vc, product);
	return product;
}

#endif /* UTILS_HPP */

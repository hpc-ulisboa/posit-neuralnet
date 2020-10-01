#ifndef QUIRE_HPP
#define QUIRE_HPP

#include <iostream>

// General headers
#include <universal/posit/posit>

// old standard
// using carry guard size = nbits-1
#if QUIRE_MODE==1

template <size_t nbits, size_t es>
using Quire = sw::unum::quire<nbits, es, nbits-2>;

// new standard
// using carry guard size = 31
#elif QUIRE_MODE==2

template <size_t nbits, size_t es>
using Quire = sw::unum::quire<nbits, es, 30>;

// not using quires
#else

template<size_t nbits, size_t es>
using Quire = sw::unum::posit<nbits, es>;

#endif /* QUIRE_MODE */

// Not using quires
#if !defined(QUIRE_MODE) || QUIRE_MODE==0

template<size_t nbits, size_t es>
inline posit<nbits, es> Quire_add(const posit<nbits, es>& lhs, const posit<nbits, es>& rhs) {
	return lhs+rhs;
}

template<size_t nbits, size_t es>
inline posit<nbits, es> Quire_mul(const posit<nbits, es>& lhs, const posit<nbits, es>& rhs) {
	return lhs*rhs;
}

// Using quires
#else

template<size_t nbits, size_t es>
inline value<nbits - es + 2> Quire_add(const posit<nbits, es>& lhs, const posit<nbits, es>& rhs) {
	return quire_add(lhs, rhs);
}

template<size_t nbits, size_t es>
inline value<2 * (nbits - 2 - es)> Quire_mul(const posit<nbits, es>& lhs, const posit<nbits, es>& rhs) {
	return quire_mul(lhs, rhs);
}

#endif /* QUIRE_MODE */

#endif /* QUIRE_HPP */

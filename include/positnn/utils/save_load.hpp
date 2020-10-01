// Based from:
// https://www.includehelp.com/code-snippets/cpp-program-to-write-and-read-an-object-in-from-a-binary-file.aspx

#ifndef SAVE_LOAD_HPP
#define SAVE_LOAD_HPP

#include <fstream>
#include <iostream>

//#include "Net.hpp"

template <typename PositFile, typename T, typename String>
int save(T& object, String filename) {
	std::ofstream file;
	file.open(filename, std::ios::out | std::ios::binary);
	
	if(!file){
		std::cout << "Error in creating file: " << filename;
		return -1;
	}

	size_t const nbits = PositFile::nbits;
	size_t const es = PositFile::es;
	file.write((char*)&nbits, sizeof(nbits));
	file.write((char*)&es, sizeof(es));

	object.template write<PositFile>(file);

	std::cout << "Saved to: " << filename << std::endl;

	return 0;
}

template <typename PositFile, typename T, typename String>
int load(T& object, String filename) {
	std::ifstream file;
	file.open(filename, std::ios::in | std::ios::binary);

	if(!file){
		std::cout << "Error in opening file: " << filename << std::endl;
		return -1;
	}
	
	size_t nbits;
	size_t es;
	file.read((char*)&nbits, sizeof(nbits));
	file.read((char*)&es, sizeof(es));
	
	if (nbits!=PositFile::nbits || es!=PositFile::es) {
		std::cerr << "given: nbits=" << PositFile::nbits << " es=" << PositFile::es << std::endl;
		std::cerr << "file: nbits=" << nbits << " es=" << es << std::endl;
		throw std::invalid_argument( "posits have different sizes" );
		return -1;
	}

	object.template read<PositFile>(file);

	std::cout << "Loaded from: " << filename << std::endl;
	
	file.close();	
	return 0;
}

#endif /* SAVE_LOAD_HPP */

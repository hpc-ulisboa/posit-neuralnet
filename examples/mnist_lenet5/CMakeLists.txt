cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnist_lenet5)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags ######################################################
set(USE_SSE OFF)
set(USE_AVX OFF)
set(USE_AVX2 OFF)

# Unix
if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR
   CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-msse3" COMPILER_HAS_SSE_FLAG)
    check_cxx_compiler_flag("-mavx"  COMPILER_HAS_AVX_FLAG)
    check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2_FLAG)

    # set Streaming SIMD Extension (SSE) instructions
	if(USE_SSE AND COMPILER_HAS_SSE_FLAG)
		set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -msse3")
	endif(USE_SSE AND COMPILER_HAS_SSE_FLAG)
    # set Advanced Vector Extensions (AVX)
	if(USE_AVX AND COMPILER_HAS_AVX_FLAG)
		set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx")
	endif(USE_AVX AND COMPILER_HAS_AVX_FLAG)
    # set Advanced Vector Extensions 2 (AVX2)
	if(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
		set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx2 -march=core-avx2")
	endif(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
endif()

add_definitions(-D LL_THREADS=4)
add_definitions(-D HL_THREADS=1)
find_package (Threads)
###########################################################################

# Compile flags ###########################################################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g -pg")
SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -pg")
###########################################################################

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_definitions(-D USING_PYTORCH)
###########################################################################

# Set source and include folder ###########################################
set(SRC_FOLDER "${CMAKE_CURRENT_LIST_DIR}/src")
set(HEADER_FOLDER "${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${HEADER_FOLDER}")
###########################################################################

# Include universal (posits) ##############################################
include_directories("${CMAKE_CURRENT_LIST_DIR}/../../universal/include")
###########################################################################

# Include PositNN (PyTorch for posits) ####################################
include_directories("${CMAKE_CURRENT_LIST_DIR}/../../include")
###########################################################################

# Setup executables #######################################################
add_executable(train_test ${SRC_FOLDER}/train_test.cpp)
target_link_libraries(train_test ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(train_test "${TORCH_LIBRARIES}")
set_property(TARGET train_test PROPERTY CXX_STANDARD 14)

###########################################################################

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnist)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags ######################################################
set(USE_SSE ON)
set(USE_AVX ON)
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

#elseif(WIN32) # MSVC
#	if(@PACKAGE_NAME@_USE_SSE)
#		add_definitions(-DCNN_USE_SSE)
#        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:SSE2")
#	endif(@PACKAGE_NAME@_USE_SSE)
#	if(@PACKAGE_NAME@_USE_AVX)
#		add_definitions(-DCNN_USE_AVX)
#        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX")
#	endif(@PACKAGE_NAME@_USE_AVX)
#	# include specific flags for release and debug modes.
#    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE}
#        /Ox /Oi /Ot /Oy /GL /fp:fast /GS- /bigobj /LTCG")
#	add_definitions(-D _CRT_SECURE_NO_WARNINGS)
endif()

add_definitions(-D LL_THREADS=1)
add_definitions(-D HL_THREADS=3)
find_package (Threads)
###########################################################################

# Compile flags ###########################################################
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O1 -g -pg")
SET(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} -pg")
###########################################################################

# Include PyTorch library (libtorch) ######################################
if (UNIX)
	if(APPLE)
		set(CMAKE_PREFIX_PATH "/home/gonced8/uni/code/libtorch-macos")
	else()
		set(CMAKE_PREFIX_PATH "/home/gonced8/uni/code/libtorch-linux")
	endif()
endif()

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
include_directories("${CMAKE_CURRENT_LIST_DIR}/../universal/include")
###########################################################################

# Include PositNN (PyTorch for posits) ####################################
include_directories("${CMAKE_CURRENT_LIST_DIR}/../positnn/include")
###########################################################################

# Setup executables #######################################################
add_executable(train_test ${SRC_FOLDER}/train_test.cpp)
target_link_libraries(train_test ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(train_test "${TORCH_LIBRARIES}")
set_property(TARGET train_test PROPERTY CXX_STANDARD 14)

add_executable(test_model ${SRC_FOLDER}/test.cpp)
target_link_libraries(test_model ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_model "${TORCH_LIBRARIES}")
set_property(TARGET test_model PROPERTY CXX_STANDARD 14)

add_executable(print_parameters ${SRC_FOLDER}/print_parameters.cpp)
target_link_libraries(print_parameters ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(print_parameters "${TORCH_LIBRARIES}")
set_property(TARGET print_parameters PROPERTY CXX_STANDARD 14)
###########################################################################

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET train
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:train>)
# endif (MSVC)

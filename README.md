# PositNN

Framework in C++ for Deep Learning (training and testing) using <a href="https://posithub.org/" target="_blank">Posits</a>.
The posits are emulated with <a href="https://github.com/stillwater-sc/universal" target="_blank">stillwater-sc/universal</a> library and the deep learning is based in <a href="https://pytorch.org/" target="_blank">PyTorch</a>.

This is being developed for a thesis to obtain a Master of Science Degree in Aerospace Engineering.

---

## Table of Contents

- [Examples](#examples)
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Tests](#tests)
- [Cite](#cite)
- [Contributing](#contributing)
- [Team](#team)
- [Support](#support)
- [License](#license)

---

## Examples

Folder "examples" includes: example of CMakeLists.txt for a project, an example of a FC Neural Network applied to the MNIST dataset, and LeNet-5 applied to the MNIST dataset. To know how to run an example, check the section [Tests](#tests).

- Example of training LeNet-5 on Fashion MNIST using posits
![training lenet-5 on mnist using posits](examples/FashionMNIST_LeNet5.png?raw=true "Example of training LeNet-5 on MNIST using posits")

---

## Installation

### Requirements

- <a href="https://github.com/gonced8/universal" target="_blank">Fork from stillwater-sc/universal: gonced8/universal</a>
- <a href="https://pytorch.org/get-started/locally/" target="_blank">PyTorch for C++ (LibTorch)</a>
- <a href="https://github.com/hpc-ulisboa/posit-neuralnet" target="_blank">PositNN (this library)</a>
- cmake >= v3
- gcc >= v5
- glibc >= v2.17

### Setup

- Clone this repository
```shell
$ git clone https://github.com/hpc-ulisboa/posit-neuralnet.git
```

- Clone gonced8/universal repository inside include folder
```shell
$ cd posit-neuralnet/include
$ git clone https://github.com/gonced8/universal.git
```

- Download appropriate PyTorch for C++ (LibTorch) and unzip also inside include folder

<a href="https://pytorch.org/get-started/locally/" target="_blank">https://pytorch.org/get-started/locally/</a>



---

## Features
- Use any posit configuration
- Activation functions: ReLU, Sigmoid, Tanh
- Layers: Batch Normalization, Convolution, Dropout, Linear (Fully-Connected), Pooling (average and max)
- Loss functions: Cross-Entropy, Mean Squared Error
- Optimizer: SGD
- Tensor class: StdTensor
- Parallelization: multithreading with std::thread

## Usage
- Copy the CMakeLists.txt inside examples and adapt to your setup, namely, the directories of universal and PositNN, and number of threads
- Build your project
```shell
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch"
$ make
```
## Tests

To check if everything is installed correctly, you can try one of the examples. The following steps describe how to test the mnist_fcnn example.

- Go to the project folder
```shell
$ cd examples/mnist_fcnn
```
- (Optional) Edit CMakeLists.txt to your configuration. Configure positnn and universal path. The one given assumes that they are both inside the include folder at the repository root directory, that is, from your current path:
```shell
../../include
../../include/universal/include
```
- Build the project. Specify absolute path to PyTorch (LibTorch) folder. This example assumes that the folder is also inside the include folder.
```shell
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH="/home/gonced8/posit-neuralnet/include/libtorch"
$ make
```

- Run program. If you're saving the models, make sure the appropriate output folder exists.
```shell
$ ./train_posit
```

---

## Cite
```bibtex
@InProceedings{Raposo2021,
  author      = {Gonçalo Raposo and Pedro Tom{\'{a}}s and Nuno Roma},
  booktitle   = {{ICASSP} 2021 - 2021 {IEEE} International Conference on Acoustics, Speech and Signal Processing ({ICASSP})},
  title       = {{PositNN: Training Deep Neural Networks with Mixed Low-Precision Posit}},
  year        = {2021},
  month       = {jun},
  pages       = {7908–7912},
  publisher   = {{IEEE}},
  abstract    = {Low-precision formats have proven to be an efficient way to reduce not only the memory footprint but also the hardware resources and power consumption of deep learning computations. Under this premise, the posit numerical format appears to be a highly viable substitute for the IEEE floating-point, but its application to neural networks training still requires further research. Some preliminary results have shown that 8-bit (and even smaller) posits may be used for inference and 16-bit for training, while maintaining the model accuracy. The presented research aims to evaluate the feasibility to train deep convolutional neural networks using posits. For such purpose, a software framework was developed to use simulated posits and quires in end-to-end training and inference. This implementation allows using any bit size, configuration, and even mixed precision, suitable for different precision requirements in various stages. The obtained results suggest that 8-bit posits can substitute 32-bit floats during training with no negative impact on the resulting loss and accuracy.},
  doi         = {10.1109/ICASSP39728.2021.9413919},
  eprint      = {2105.00053},
  eprintclass = {cs.LG},
  eprinttype  = {arXiv},
  keywords    = {posit numerical format, low-precision arithmetic, deep neural networks, training, inference},
  url         = {https://ieeexplore.ieee.org/document/9413919},
}
```

## Contributing

If you'd like to get involved, e-mail me at gonced8@gmail.com

---

## Team
### Student
- Gonçalo Eduardo Cascalho Raposo - <a href="https://github.com/gonced8" target="_blank">@gonced8</a>

### Supervisors
- Prof. Nuno Roma
- Prof. Pedro Tomás - <a href="https://github.com/pedrotomas" target="_blank">@pedrotomas</a>
---

## Support

Reach out to me at: gonced8@gmail.com

---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- README.md based from <a href="https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46" target="_blank">FVCproductions</a>.

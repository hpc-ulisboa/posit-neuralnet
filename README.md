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

- <a href="https://github.com/stillwater-sc/universal" target="_blank">stillwater-sc/universal</a>
- <a href="https://pytorch.org/get-started/locally/" target="_blank">PyTorch for C++ (LibTorch)</a>
- <a href="https://github.com/hpc-ulisboa/posit-neuralnet" target="_blank">PositNN (this library)</a>
- cmake >= v3
- gcc >= v5
- glibc >= v2.17

### Setup

- Clone stillwater-sc/universal repository
```shell
$ git clone https://github.com/stillwater-sc/universal.git
```

- Download appropriate PyTorch for C++ (LibTorch) and unzip

<a href="https://pytorch.org/get-started/locally/" target="_blank">https://pytorch.org/get-started/locally/</a>

- Clone this repository
```shell
$ git clone https://github.com/hpc-ulisboa/posit-neuralnet.git
```

---

## Features
- Use any posit configuration
- Activation functions: ReLU, Sigmoid, Tanh
- Layers: Batch Normalization, Convolution, Fully-Connected, Pooling (average and max)
- Loss functions: Cross-Entropy, Mean Squared Error
- Optimizer: SGD
- Tensor class: StdTensor
- Multithreading

## Usage
- Copy the CMakeLists.txt inside examples and adapt to your setup, namely, the directories of universal and PositNN, and number of threads
- Build your project
```shell
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH="../../libtorch"
$ make
```
## Tests

To check if everything is installed correctly, you can try one of the examples. The following steps describe how to test the mnist_lenet5 example.

- Go to the project folder
```shell
$ cd examples/mnist_lenet5
```
- (Optional) Edit CMakeLists.txt to your configuration. Configure positnn and universal path. The one given assumes that universal folder is at the repository root directory, that is, from your current path:
```shell
../../include
../../universal/include
```
- Build project. Specify absolute path to PyTorch (LibTorch) folder. This example assumes that the folder is at the repository root directory.
```shell
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH="/home/gonced8/posit-neuralnet/libtorch"
$ make
```

- Run program. If you're saving the models, make sure the appropriate output folder exists.
```shell
$ ./train_test
```

---

## Contributing

If you'd like to get involved, e-mail me at gonced8@gmail.com

---

## Team
### Student
- Gonçalo Eduardo Cascalho Raposo - @gonced8

### Supervisors
- Prof. Nuno Roma
- Prof. Pedro Tomás - @pedrotomas
---

## Support

Reach out to me at: gonced8@gmail.com

---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- README.md based from <a href="https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46" target="_blank">FVCproductions</a>.

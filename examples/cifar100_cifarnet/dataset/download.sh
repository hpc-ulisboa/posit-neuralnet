#!/bin/bash

# Download files
curl https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz --output cifar-100-binary.tar.gz

# Uncompress files
tar -zxvf cifar-100-binary.tar.gz

# Move from folder
mv cifar-100-binary/* .

# Delete unnecessary
rm cifar-100-binary.tar.gz
rm fine_label_names.txt~
rm -r cifar-100-binary

// Downloaded from ChipsSpectre/chipstorch-yolov3
// https://github.com/ChipsSpectre/chipstorch-yolov3

#ifndef CIFAR10DATA_HPP
#define CIFAR10DATA_HPP

#include <torch/torch.h>

/**
 * Implementation of a cifar10 data loader.
 */
class Cifar10Data : public torch::data::Dataset<Cifar10Data> {
private:
    // number of training images
    int countTrain = 50000;
    // number of test images
    int countTest = 10000;
    int kImageRows = 32;
    int kImageColumns = 32;
    int kColorChannels = 3;
    int ENTRY_LENGTH = 3073;
    int countBatch = 10000;

    /**
     * Joins two paths. They are composed such that the return
     * value can be used for path specifications.
     * @param head - first part
     * @param tail - new, second part
     * @return combination of first and second part.
     */
    std::string join_paths(std::string head, const std::string& tail) {
        if (head.back() != '/') {
            head.push_back('/');
        }
        head += tail;
        return head;
    }

    /**
     * Loads a tensor from a binary cifar10 batch file.
     *
     * Contains all data, i.e. images and targets of the batch.
     *
     * @param full_path - full path to the filename of the batch file.
     * @return a tensor that contains count_batch x ENTRY_LENGTH items
     */
    torch::Tensor loadTensorFromBatch(const std::string& full_path) {
        auto tensor =
                torch::empty({countBatch, ENTRY_LENGTH}, torch::kByte);
        std::ifstream images(full_path, std::ios::binary);
        //AT_CHECK(images, "Error opening images file at ", full_path);
        images.read(reinterpret_cast<char *>(tensor.data_ptr()), tensor.numel());
        return tensor;
    }

    torch::Tensor readImages(const std::string& path, bool isTraining) {
        if(isTraining) {
            std::vector<std::string> paths = {"data_batch_1.bin", "data_batch_2.bin",
                                              "data_batch_3.bin", "data_batch_4.bin",
                                              "data_batch_5.bin"};
            auto trainTensor = torch::empty({5, countBatch, ENTRY_LENGTH - 1});
            for(unsigned int i = 0; i<paths.size(); i++) {
                auto currPath = paths[i];
                auto currTensor = loadTensorFromBatch(join_paths(path, currPath));

                auto currIdx = torch::empty({ENTRY_LENGTH - 1}, torch::kLong);
                for (int j = 0; j < ENTRY_LENGTH - 1; j++) {
                    currIdx[j] = j + 1;
                }
                currTensor = currTensor.index_select(1, currIdx);

                trainTensor[i] = currTensor;
            }
            trainTensor = trainTensor.reshape({countTrain, ENTRY_LENGTH - 1});
            return trainTensor.reshape({countTrain, kColorChannels, kImageRows, kImageColumns}).to(torch::kFloat32).div_(255);
        } else {
            auto tensor = loadTensorFromBatch(join_paths(path, "test_batch.bin"));

            auto idx = torch::empty({ENTRY_LENGTH - 1}, torch::kLong);
            for (int i = 0; i < ENTRY_LENGTH - 1; i++) {
                idx[i] = i + 1;
            }
            tensor = tensor.index_select(1, idx);
            return tensor.reshape({countTest, kColorChannels, kImageRows, kImageColumns}).to(torch::kFloat32).div_(255);
        }
    }

    /**
     * Reads the targets.
     * @param path - path to the cifar10 dataset
     * @param isTraining - specifies if training should be activated or not
     * @return tensor containing all required target labels, 1-dimensional
     */
    torch::Tensor readTargets(const std::string& path, bool isTraining) {
        if(isTraining) {
            std::vector<std::string> paths = {"data_batch_1.bin", "data_batch_2.bin",
                                              "data_batch_3.bin", "data_batch_4.bin",
                                              "data_batch_5.bin"};
            auto trainTensor = torch::empty({5, countBatch});
            for(unsigned int i = 0; i<paths.size(); i++) {
                auto currPath = paths[i];
                auto currTensor = loadTensorFromBatch(join_paths(path, currPath));

                auto idx = torch::full({1}, 0, torch::kLong);
                trainTensor[i]  = currTensor.index_select(1, idx).reshape({countBatch});
            }

            return trainTensor.reshape({countTrain}).to(torch::kLong);
        } else {
            auto tensor = loadTensorFromBatch(join_paths(path, "test_batch.bin"));

            auto idx = torch::full({1}, 0, torch::kLong);
            tensor = tensor.index_select(1, idx);
            return tensor.reshape({countTest}).to(torch::kLong);
        }
    }
public:
    Cifar10Data(const std::string& path, bool isTraining)
        : _isTraining(isTraining),
          _images(readImages(path, isTraining)),
          _targets(readTargets(path, isTraining))
    {

    }

    torch::data::Example<> get(size_t index) override {
        return {_images[index], _targets[index]};
    }

    c10::optional<size_t> size() const override {
        if(_isTraining) {
            return countTrain;
        }
        return countTest;
    }
private:
    bool _isTraining;
    torch::Tensor _images, _targets;
};

#endif /* CIFAR10DATA_HPP */

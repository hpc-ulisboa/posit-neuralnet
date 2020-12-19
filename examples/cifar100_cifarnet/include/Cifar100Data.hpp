// Based from the code downloaded from ChipsSpectre/chipstorch-yolov3
// https://github.com/ChipsSpectre/chipstorch-yolov3

#ifndef CIFAR100DATA_HPP
#define CIFAR100DATA_HPP

#include <torch/torch.h>

/**
 * Implementation of a cifar10 data loader.
 */
class Cifar100Data : public torch::data::Dataset<Cifar100Data> {
private:
    // number of training images
    int countTrain = 50000;
    // number of test images
    int countTest = 10000;
    int kImageRows = 32;
    int kImageColumns = 32;
    int kColorChannels = 3;
    int ENTRY_LENGTH = 3074;

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
    torch::Tensor loadTensorFromBatch(const std::string& full_path, int count) {
        auto tensor =
                torch::empty({count, ENTRY_LENGTH}, torch::kByte);
        std::ifstream images(full_path, std::ios::binary);
        //AT_CHECK(images, "Error opening images file at ", full_path);
        images.read(reinterpret_cast<char *>(tensor.data_ptr()), tensor.numel());
        return tensor;
    }

    torch::Tensor readImages(const std::string& path, bool isTraining) {
		std::string filename = (isTraining) ? "train.bin" : "test.bin";
		int count = (isTraining) ? countTrain : countTest;

		auto tensor = loadTensorFromBatch(join_paths(path, filename), count);

		auto idx = torch::empty({ENTRY_LENGTH - 2}, torch::kLong);
		for (int i=0; i<ENTRY_LENGTH-2; i++) {
			idx[i] = i + 2;
		}
		tensor = tensor.index_select(1, idx);

		return tensor.reshape({count, kColorChannels, kImageRows, kImageColumns}).to(torch::kFloat32).div_(255);
    }

    /**
     * Reads the targets.
     * @param path - path to the cifar10 dataset
     * @param isTraining - specifies if training should be activated or not
     * @return tensor containing all required target labels, 1-dimensional
     */
    torch::Tensor readTargets(const std::string& path, bool isTraining, bool fine_label) {
		std::string filename = (isTraining) ? "train.bin" : "test.bin";
		int count = (isTraining) ? countTrain : countTest;
		int label_idx = (fine_label) ? 1 : 0;

		auto tensor = loadTensorFromBatch(join_paths(path, filename), count);

		auto idx = torch::full({1}, label_idx, torch::kLong);
		tensor = tensor.index_select(1, idx);

		return tensor.reshape({count}).to(torch::kLong);
    }
public:
    Cifar100Data(const std::string& path, bool isTraining, bool fine_label)
        : _isTraining(isTraining),
		  _fine_label(fine_label),
          _images(readImages(path, isTraining)),
          _targets(readTargets(path, isTraining, fine_label))
    { }

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
	bool _fine_label;
    torch::Tensor _images, _targets;
};

#endif /* CIFAR100DATA_HPP */

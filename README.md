# INQ-incremental-network-quantization-towards-lossless-CNNs-with-low-precision-weights

论文地址：https://arxiv.org/abs/1702.03044
INQ作者的caffe代码实现：https://github.com/Zhouaojun/Incremental-Network-Quantization

本实现是基于TensorFlow，数据集是cifar-10，网络模型是ResNet。分别在ResNet-20、ResNet-32、ResNet-44以及ResNet-56上进行了实验。
量化的位数分别是3bits、4bits、5bits，但量化3bits时，模型并不收敛，可能在某些细节上的复现出了问题。

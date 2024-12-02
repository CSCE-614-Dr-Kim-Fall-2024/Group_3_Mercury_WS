MERCURY: Accelerating DNN Training by Exploiting Input Similarity
Overview
This project implements a software-based adaptation of the MERCURY framework, originally proposed as a hardware accelerator for deep neural network (DNN) training. MERCURY utilizes Random Projection with Quantization (RPQ) and a specialized caching mechanism (MCACHE) to optimize training by reusing computations for similar inputs. Our implementation adapts this concept to four DNN architectures:

AlexNet
VGG13
VGG16
A custom-designed CNN model
All experiments are conducted on the CIFAR-10 dataset, with the primary goal of achieving significant training speedups while maintaining model accuracy.

Features
RPQ Mechanism: Efficiently transforms input vectors into binary signatures, enabling fast similarity detection.
MCACHE System: Stores computed results for reuse, minimizing redundant computations.
Scalable Design: Adapted and implemented on multiple DNN architectures.
Performance Analysis: Demonstrates speedups in training across models.
Dataset
The experiments utilize the CIFAR-10 dataset, comprising 60,000 32x32 color images in 10 classes:

Training Set: 50,000 images
Testing Set: 10,000 images
For enhanced generalization, a subset of the 80 Million Tiny Images dataset is used for augmentation.

Implementation Details
Custom CNN:

Three convolutional layers
Max pooling
Two fully connected layers
Optimizations:

RPQ reduces the dimensionality of inputs while preserving similarity.
Binary signatures stored in MCACHE to skip redundant convolutions.
Directly mapped cache structure with hit/miss tracking.
Integration:

Extended RPQ and MCACHE to AlexNet, VGG13, and VGG16 models.
GPU parallelization simulates the hardware-level benefits described in the MERCURY paper.
System Specifications:

NVIDIA RTX 4060 GPU
Varying clock speeds for different models (450 MHz to 2555 MHz).
Results
Custom Model: Achieved a speedup of 1.13× with the MERCURY-inspired design.
AlexNet, VGG13, VGG16: Demonstrated increasing speedups with larger architectures.
Adaptability: Dynamic toggling of RPQ and MCACHE mechanisms across layers to optimize performance.

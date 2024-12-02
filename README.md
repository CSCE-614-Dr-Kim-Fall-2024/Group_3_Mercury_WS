# MERCURY: Accelerating DNN Training by Exploiting Input Similarity

## Overview

This project implements a software-based adaptation of the **MERCURY** framework, originally proposed as a hardware accelerator for deep neural network (DNN) training. MERCURY utilizes **Random Projection with Quantization (RPQ)** and a specialized caching mechanism (MCACHE) to optimize training by reusing computations for similar inputs. Our implementation adapts this concept to four DNN architectures:

- **AlexNet**
- **VGG13**
- **VGG16**
- A custom-designed CNN model

All experiments are conducted on the CIFAR-10 dataset, with the primary goal of achieving significant training speedups while maintaining model accuracy.

---

## Features

- **RPQ Mechanism**: Efficiently transforms input vectors into binary signatures, enabling fast similarity detection.
- **MCACHE System**: Stores computed results for reuse, minimizing redundant computations.
- **Scalable Design**: Adapted and implemented on multiple DNN architectures.
- **Performance Analysis**: Demonstrates speedups in training across models.

---

## Dataset

The experiments utilize the **CIFAR-10** dataset, comprising 60,000 32x32 color images in 10 classes:
- **Training Set**: 50,000 images
- **Testing Set**: 10,000 images

For enhanced generalization, a subset of the **80 Million Tiny Images** dataset is used for augmentation.

---

## Implementation Details

1. **Custom CNN**:
    - Three convolutional layers
    - Max pooling
    - Two fully connected layers

2. **Optimizations**:
    - RPQ reduces the dimensionality of inputs while preserving similarity.
    - Binary signatures stored in MCACHE to skip redundant convolutions.
    - Directly mapped cache structure with hit/miss tracking.

3. **Integration**:
    - Extended RPQ and MCACHE to AlexNet, VGG13, and VGG16 models.
    - GPU parallelization simulates the hardware-level benefits described in the MERCURY paper.

4. **System Specifications**:
    - NVIDIA RTX 4060 GPU
    - Varying clock speeds for different models (450 MHz to 2555 MHz).

---

## Results

- **Custom Model**: Achieved a speedup of **1.13×** with the MERCURY-inspired design.
- **AlexNet, VGG13, VGG16**: Demonstrated increasing speedups with larger architectures.
- **Adaptability**: Dynamic toggling of RPQ and MCACHE mechanisms across layers to optimize performance.

---

## Usage

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run experiments:
    - For the custom CNN model:
        ```bash
        python run_custom_model.py
        ```
    - For AlexNet:
        ```bash
        python run_alexnet.py
        ```
    - For VGG13:
        ```bash
        python run_vgg13.py
        ```
    - For VGG16:
        ```bash
        python run_vgg16.py
        ```

4. View results:
    - Training logs and performance metrics are stored in the `results/` directory.

---

## Project Structure

```plaintext
.
├── datasets/             # Dataset handling scripts
├── models/               # DNN model architectures
├── results/              # Output logs and metrics
├── scripts/              # Scripts for running experiments
├── utils/                # Helper functions (e.g., RPQ, caching mechanisms)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```
## References

- **Research Paper**: [MERCURY: Accelerating DNN Training by Exploiting Input Similarity](https://doi.org/your-paper-doi)
- **Dataset**: [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Framework**: [PyTorch Official Site](https://pytorch.org/)
- **GPU Specifications**: [NVIDIA Official Site](https://www.nvidia.com/)

---

## Contributors

### **Vishwa Raj**
- Custom model design and RPQ (Relaxed Priority Queue) implementation.

### **Hitarth**
- Dataset preparation and experimental analysis.

### **Kowsalya**
- Integration with AlexNet, VGG13, and VGG16 models.

### **Poornima Gulur Chakrapani**
- Performance evaluation and report documentation.

---

# Group_3_Mercury_WS

RPQ and WS Dataflow Implementation for ResNet-50
This repository implements Random Projection and Quantization (RPQ) and Weight-Stationary (WS) Dataflow on a ResNet-50 model using the TinyImageNet dataset. The goal is to optimize training by reducing redundant computations and managing memory efficiently without modifying the model structure.

Features
1. RPQ as a Separate Function
The RPQ process (rpq_function) is implemented independently of the model architecture.
It utilizes:
A random projection matrix to generate signatures for inputs.
An mcache to store outputs based on RPQ signatures, avoiding redundant computations.
2. Memory Optimization
Batch Size:

Reduced batch size to prevent CUDA out-of-memory errors.
Default: batch_size=16 (can be adjusted based on available GPU memory).
mcache:

Fixed-size cache (mcache_limit=5000) to control GPU memory usage.
Cache stores results of computations for reuse during training.
3. No Model Modifications
The original ResNet-50 architecture is retained without any structural changes.
RPQ and WS are applied externally during training, ensuring compatibility with any standard model.
4. mcache Handling
Cache Miss:

When an input signature is not found in mcache, the computation proceeds as normal.
The result is stored in mcache for future reuse, provided the cache size limit is not exceeded.
Cache Hit:

If an input signature is found in mcache, the cached output is used, skipping redundant computation.
How It Works
RPQ (Random Projection and Quantization)
Each input tensor is projected using a random matrix to generate binary signatures.
These signatures are used as keys to store and retrieve cached outputs from mcache.
WS (Weight-Stationary Dataflow)
Weights for convolutional layers are cached after the first forward pass for reuse.
This minimizes redundant weight movements during training.

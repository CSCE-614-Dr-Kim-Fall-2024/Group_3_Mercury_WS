import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from time import perf_counter

CLOCK_SPEED = 2555  # (MHz) Average speed of my RTX 4060

# 1. Hyperparameters
batch_size = 1
learning_rate = 0.001
num_epochs = 3

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Data Preprocessing: Convert to grayscale with 1 channel and normalize
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with 1 channel
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# 3. Load CIFAR-10 Dataset
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)

# 4. Define AlexNet Model for 1-channel Grayscale Input
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Grayscale input, (3x32x32 -> 64x32x32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x32x32 -> 64x16x16
            
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),  # 64x16x16 -> 192x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 192x16x16 -> 192x8x8
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 192x8x8 -> 384x8x8
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 384x8x8 -> 256x8x8
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256x8x8 -> 256x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 256x8x8 -> 256x4x4
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten to (batch_size, 256*4*4)
            nn.Linear(256 * 4 * 4, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            
            nn.Linear(2048, 10) # 10 classes
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Initialize AlexNet model
model = AlexNet().to(device)

# 5. RPQ Function
def rpq(input_batch, rows, columns):
    # Flatten each image in the batch
    flattened_batch = input_batch.view(input_batch.size(0), -1)  # Shape: (batch_size, rows)
    # Perform matrix multiplication in parallel for the batch
    signature = torch.matmul(flattened_batch, random_rpq_matrix)  # Parallel dot product
    # Quantization
    signature_quantized = torch.where(signature < 0, torch.ones_like(signature), torch.zeros_like(signature))
    return signature_quantized

# 6. Training Loop with CUDA Timing
sync = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    cache_hits = 0
    cache_misses = 0
    mcache = []
    total_rpq = 0
    random_rpq_matrix = torch.randn(1024, 20, device=device).uniform_(-1,1) # Move RPQ matrix to GPU #mean = 0 and var = 1
    start = perf_counter()
    # Forward pass
    for input_images, _ in train_loader:
        input_images = input_images.to(device)  # Move batch to GPU

        # Start RPQ computation
        start_rpq = perf_counter()
        rpq_signature_output = rpq(input_images, 1024, 20)  # Parallel RPQ for the entire batch
        torch.cuda.synchronize()  # Synchronize GPU threads after RPQ computation
        end_rpq = perf_counter()

        # Generate binary keys for the entire batch
        binary_keys = [''.join(map(str, row.int().tolist())) for row in rpq_signature_output]
        total_rpq += end_rpq - start_rpq

        # Sequential cache mechanism
        for binary_key in binary_keys:
            if binary_key in mcache:
                cache_hits += 1
            else:
                cache_misses += 1
                model(input_images)  # Pass the batch or individual image here as needed
                mcache.append(binary_key)    
    
    #sycnrhonize cuda cycles
    torch.cuda.synchronize()
    
    end = perf_counter()
    sync += (end - start)
    # Compute elapsed time
    print(f"Cache_hits:{cache_hits}")
    print(f"Cache_misses:{cache_misses}")
    print(f"TOTAL TIME TAKEN in EACH EPOCH: {end - start} s")
    print(f"TOTAL RPQ TIME TAKEN in EACH EPOCH: {total_rpq} s")

print(f"Total time: {sync}s")
print(f"CYCLES: {sync * CLOCK_SPEED} * 10^6")
print("Training complete!")
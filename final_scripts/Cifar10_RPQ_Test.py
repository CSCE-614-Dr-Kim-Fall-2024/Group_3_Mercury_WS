import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from time import perf_counter

CLOCK_SPEED = 450  #(MHz) (Average speed of my RTX 4060)

# 1. Hyperparameters
batch_size = 1
learning_rate = 0.001
num_epochs = 10 

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 2. Data Preprocessing: No resizing, keep original 32x32 size
transform = transforms.Compose([
      transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
      transforms.ToTensor(),                        # Convert to tensor
      transforms.Normalize(mean=[0.5], std=[0.5])   # Normalize to [-1, 1]
])

# 3. Load CIFAR-10 Dataset
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 6. Data Loaders
train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)

##########################################################################################################################

# RPQ Function
def rpq(input_vector, rows, columns):
    flattened_vector = input_vector.view(input_vector.size(0),-1)
    # Dot product of input vector and R
    signature = torch.matmul(flattened_vector, random_rpq_matrix)

    # Quantization -> sign-based
    signature_quantized = torch.where(signature < 0, torch.ones_like(signature), torch.zeros_like(signature))
    return signature_quantized

# 7. Define a Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 classes: automobile and dog

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
model = SimpleCNN().to(device)  # Move model to GPU

# 8. Training Loop with CUDA Timing
sync = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    cache_hits = 0
    cache_misses = 0
    mcache = []
    total_rpq = 0
    random_rpq_matrix = torch.randn(1024, 20, device=device).uniform_(-1,1) # Move RPQ matrix to GPU #mean = 0 and var = 1
    start = perf_counter()
    # Forward pass
    for input_image, _ in train_loader:
        input_image = input_image.to(device)
        start_rpq = perf_counter()
        rpq_signature_output = rpq(input_image, 1024, 20)# 1024 coz 32 * 32 (right now) , 30 columns coz signature length
        end_rpq = perf_counter()
        binary_key = ''.join(map(str, rpq_signature_output.int().tolist()))  # tensor to list and then to string
        total_rpq += end_rpq - start_rpq
        #cache mechanism
        if binary_key in mcache:
            cache_hits += 1
        else:
            cache_misses += 1
            model(input_image)
            mcache.append(binary_key)
        torch.cuda.synchronize()
    end = perf_counter()
    sync += (end - start)

    # Compute elapsed time
    print(f"Cache_hits:{cache_hits}")
    print(f"Cache_misses:{cache_misses}")
    print(f"TOTAL TIME TAKEN in EACH EPOCH: {end - start} s")
    print(f"TOTAL RPQ TIME TAKEN in EACH EPOCH: {total_rpq} s")

print(f"Total time:{sync}s")
print(f"CYCLES: {sync * CLOCK_SPEED}*10e6")
print("Training complete!")

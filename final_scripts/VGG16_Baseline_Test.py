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
#full_test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader for CIFAR-10 (filtered)
train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
#test_loader = torch.utils.data.DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False)

# 4. Define VGG16 Model from Scratch for 1 channel grayscale input
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # Convolutional layers with max pooling
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 1 * 1, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 10)  # 10 classes

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.max_pool2d(x, 2)

        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.max_pool2d(x, 2)

        x = torch.relu(self.conv7(x))
        x = torch.relu(self.conv8(x))
        x = torch.max_pool2d(x, 2)

        x = torch.relu(self.conv9(x))
        x = torch.relu(self.conv10(x))
        x = torch.max_pool2d(x, 2)

        x = x.view(-1, 512 * 1 * 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize VGG16 model
model = VGG16().to(device)

# 7. Training Loop with CUDA Timing
sync = 0
start = 0
end = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    start = perf_counter()
    # Forward pass
    for input_images, _ in train_loader:
        input_images = input_images.to(device)  # Move batch to GPU
        model(input_images)  # Pass the batch or individual image here as needed
    end = perf_counter()
    
    print(f"TOTAL TIME TAKEN in EACH EPOCH: {end - start} s")
    sync += (end - start)
print(f"Total time:{sync}s")
print(f"CYCLES: {sync * CLOCK_SPEED}*10e6")
print("Training complete!")
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
    transforms.RandomHorizontalFlip(p=0.5),       # Randomly flip the image horizontally with a probability of 50%
    transforms.RandomRotation(degrees=15),        # Randomly rotate the image within ±15 degrees
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),  # Random crop and resize back to 32x32
    transforms.ColorJitter(brightness=0.2, contrast=0.2),      # Randomly adjust brightness and contrast
    transforms.ToTensor(),                        # Convert to tensor
])

# 3. Load CIFAR-10 Dataset
full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 4. Data Loaders
train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
############################################################################################################

# 5. Define a Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)  # Move model to GPU

# 6. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 7. Training Loop with CUDA Timing
sync = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    start = perf_counter()
    # Forward pass
    for input_image, input_label in train_loader:
        input_image, input_label = input_image.to(device), input_label.to(device)
        outputs = model(input_image)
        
    end = perf_counter()
    sync += (end - start)

    # Compute elapsed time
    print(f"TOTAL TIME TAKEN in EACH EPOCH: {end - start} s")

print(f"Total time:{sync}s")
print(f"CYCLES: {sync * CLOCK_SPEED}*10e6")
print("Training complete!")
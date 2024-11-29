import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Step 1: Data Preparation
def load_data(file_zero, file_one):
    """Loads and normalizes data from files."""
    def read_file(file_path):
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            sample = []
            for line in lines:
                if line.strip():  # If the line is not blank
                    sample.append([float(x) for x in line.split()])
                    if len(sample) == 8:  # If we have 8 lines, it's a complete sample
                        data.append(sample)
                        sample = []
        return np.array(data).reshape(len(data), -1)  # Flatten each sample to a single row

    data_zero = read_file(file_zero)
    data_one = read_file(file_one)
    all_data = np.vstack((data_zero, data_one))
    return all_data

# Load data
file_zero_PATH = "./dataset/0_dataset.dat"
file_one_PATH  = "./dataset/1_dataset.dat"
all_data = load_data(file_zero_PATH, file_one_PATH)

# Define custom input distributions
N = 8
custom_input_distributions = np.random.normal(loc=0.5, scale=0.1, size=(all_data.shape[0], N))  # Example Gaussian input distributions
custom_input_distributions = np.clip(custom_input_distributions, 0.0001, 1)  # Ensure values are between 0 and 1

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(
    custom_input_distributions, all_data, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
x_train = torch.tensor(x_train, dtype=torch.float32).cuda()
y_train = torch.tensor(y_train, dtype=torch.float32).cuda()
x_val = torch.tensor(x_val, dtype=torch.float32).cuda()
y_val = torch.tensor(y_val, dtype=torch.float32).cuda()

# Step 2: Define the Model
class GenerativeFFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(GenerativeFFNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return torch.softmax(self.network(x), dim=1)  # Normalize output as a probability distribution

# Model Parameters
input_size = x_train.size(1)
hidden_size = 20
output_size = 8
num_layers = 4

model = GenerativeFFNN(input_size, hidden_size, output_size, num_layers).cuda()

# Step 3: Training Setup
loss_fn = nn.KLDivLoss(reduction="batchmean")  # Compare probability distributions
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sanity check
print(f"x_train contains NaN: {torch.isnan(torch.log(x_train)).any()}")
print(f"y_train contains NaN: {torch.isnan(torch.log(y_train)).any()}")
print(f"x_train contains Inf: {torch.isinf(torch.log(x_train)).any()}")
print(f"y_train contains Inf: {torch.isinf(torch.log(y_train)).any()}")

with torch.no_grad():
    test_output = model(x_train[:32])
    print(f"Initial model output: {test_output}")
    print(f"Output contains NaN: {torch.isnan(test_output).any()}")
    print(f"Output min: {test_output.min()}, max: {test_output.max()}")

    print(f"Inital y_train: {y_train[:32]}")
    print(f"y_train contains NaN: {torch.isnan(y_train[:32]).any()}")
    print(f"y_train min: {y_train.min()}, max: {y_train.max()}")
    
initial_loss = loss_fn(torch.log(test_output), torch.log(y_train[:32] + 1e-8)).item()
print(f"Initial loss: {initial_loss}")


# Step 4: Training Loop
epochs = 10
batch_size = 32
train_size = x_train.size(0)
training_loss_values = []
validation_loss_values = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for i in range(0, train_size, batch_size):
        # Batch data
        x_batch = x_train[i : i + batch_size]
        y_batch = y_train[i : i + batch_size]
        
        # Forward pass
        outputs = torch.log(torch.clamp(model(x_batch),min=1e-8))  # Log of predicted distribution because torch.KLDivLoss expects
        target = torch.log(y_batch + 1e-8)   # log-probabilities as input and normal probabilities as target
        
        # Compute loss
        loss = loss_fn(outputs, target)
        training_loss_values.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = torch.log(model(x_val))
        val_target = torch.log(y_val + 1e-8)
        val_loss = loss_fn(val_outputs, val_target).item()
        validation_loss_values.append(val_loss)
    
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss/train_size:.6f}, Val Loss: {val_loss:.6f}")
    print(f"Output contains NaN: {torch.isnan(torch.log(model(x_batch))).any()}")
    print(f"Target contains NaN: {torch.isnan(torch.log(y_batch)).any()}")

print("Training Complete!")

# Plot the training and validation loss and save
plt.plot(training_loss_values, label="Training Loss")
plt.plot(validation_loss_values, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")
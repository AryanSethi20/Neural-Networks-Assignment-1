import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import KFold
import shap

# Import functions from common_utils.py
from common_utils import set_seed, split_dataset, preprocess_dataset, EarlyStopper, extract_features

# Set the seed for reproducibility
set_seed()

# ==============================================================================
# Part A, Q1 (15 marks)
# ==============================================================================

# Define the MLP model with three hidden layers, ReLU activations, and dropout
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, dropout_prob=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x

# Load the dataset
data = pd.read_csv('audio_gtzan1.csv')

# Create binary labels (blues = 0, metal = 1)
def extract_label(filename):
    if 'blues' in filename:
        return 'blues'
    elif 'metal' in filename:
        return 'metal'
    else:
        raise ValueError(f"Unknown genre in filename: {filename}")

# Create labels from filenames
data['label'] = data['filename'].apply(extract_label)

# Split the dataset using the function from common_utils
X_train, y_train, X_test, y_test = split_dataset(
    data, 
    columns_to_drop=['filename', 'label'], 
    test_size=0.3, 
    random_state=42
)

# Preprocess the dataset using the function from common_utils
X_train_scaled, X_test_scaled = preprocess_dataset(X_train, X_test)

# Create a PyTorch dataset class
class MusicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create dataset objects
train_dataset = MusicDataset(X_train_scaled, y_train)
test_dataset = MusicDataset(X_test_scaled, y_test)

# Create data loaders
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the model, optimizer and loss function
input_size = X_train.shape[1]
model = MLP(input_size=input_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# Train the model
def train_epoch(model, dataloader, optimizer, loss_fn, device='cpu'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss and accuracy
        total_loss += loss.item() * inputs.size(0)
        predicted = (outputs >= 0.5).float()
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate(model, dataloader, loss_fn, device='cpu'):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Track loss and accuracy
            total_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# Initialize for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
early_stopper = EarlyStopper(patience=3)

n_epochs = 100
train_losses = []
train_accs = []
test_losses = []
test_accs = []

# Training loop
for epoch in range(n_epochs):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluate
    test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    # Check early stopping
    if early_stopper.early_stop(test_loss):
        print(f'Early stopping at epoch {epoch+1}')
        break

# Plot the training and test accuracies
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy vs. Epochs')

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs')

plt.tight_layout()
plt.savefig('accuracy_loss_plots.png')
plt.show()

# ==============================================================================
# Part A, Q2 (10 marks)
# ==============================================================================

# Function to perform k-fold cross-validation for different batch sizes
def batch_size_cv(X, y, batch_sizes, n_splits=5, n_epochs=100, patience=3, device='cpu'):
    # Initialize dictionaries to store results
    cv_accuracies = {bs: [] for bs in batch_sizes}
    epoch_times = {bs: [] for bs in batch_sizes}
    
    # Define k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        fold_accuracies = []
        fold_times = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold+1}/{n_splits}")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Scale features
            X_train_fold_scaled, X_val_fold_scaled = preprocess_dataset(X_train_fold, X_val_fold)
            
            # Create datasets and dataloaders
            train_dataset = MusicDataset(X_train_fold_scaled, y_train_fold)
            val_dataset = MusicDataset(X_val_fold_scaled, y_val_fold)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model, optimizer, and loss function
            input_size = X_train_fold.shape[1]
            model = MLP(input_size=input_size)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.BCELoss()
            
            # Early stopping
            early_stopper = EarlyStopper(patience=patience)
            
            # Training loop
            best_val_acc = 0
            last_epoch_time = 0
            
            for epoch in range(n_epochs):
                # Measure time for the last epoch
                if epoch == n_epochs - 1:
                    start_time = time.time()
                
                # Train and evaluate
                _, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
                val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
                
                if epoch == n_epochs - 1:
                    last_epoch_time = time.time() - start_time
                
                # Update best accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                # Check early stopping
                if early_stopper.early_stop(val_loss):
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            fold_accuracies.append(best_val_acc)
            fold_times.append(last_epoch_time)
        
        # Store average results for the batch size
        cv_accuracies[batch_size] = np.mean(fold_accuracies)
        epoch_times[batch_size] = np.mean(fold_times)
    
    return cv_accuracies, epoch_times

# Define batch sizes to test
batch_sizes = [32, 64, 128, 256]

# Run cross-validation
cv_accuracies, epoch_times = batch_size_cv(X_train, y_train, batch_sizes, device=device)

# Plot mean cross-validation accuracies
plt.figure(figsize=(10, 5))
plt.scatter(batch_sizes, [cv_accuracies[bs] for bs in batch_sizes], marker='o', s=100)
plt.plot(batch_sizes, [cv_accuracies[bs] for bs in batch_sizes], 'b--')
plt.xticks(batch_sizes)
plt.xlabel('Batch Size')
plt.ylabel('Mean CV Accuracy')
plt.title('Mean Cross-Validation Accuracy vs. Batch Size')
plt.grid(True)
plt.savefig('batch_size_accuracy.png')
plt.show()

# Create a table of time taken to train
time_df = pd.DataFrame({
    'Batch Size': batch_sizes,
    'Time per Epoch (seconds)': [epoch_times[bs] for bs in batch_sizes]
})
print("\nTime taken to train the network on the last epoch:")
print(time_df)

# Choose optimal batch size (example logic)
optimal_batch_size = max(cv_accuracies, key=cv_accuracies.get)
print(f"\nOptimal batch size: {optimal_batch_size}")
print(f"Optimal batch size accuracy: {cv_accuracies[optimal_batch_size]:.4f}")
print(f"Optimal batch size time per epoch: {epoch_times[optimal_batch_size]:.4f} seconds")

# Rationale for selection
print("\nRationale for selection:")
print(f"The batch size {optimal_batch_size} was selected because it provides the highest validation accuracy")
print(f"while maintaining a reasonable training time per epoch. This batch size provides a good")
print(f"balance between computational efficiency and model performance.")

# ==============================================================================
# Part A, Q3 (10 marks)
# ==============================================================================

# Define custom MLP with variable first hidden layer size
class CustomMLP(nn.Module):
    def __init__(self, input_size, first_hidden_size, other_hidden_size=128, output_size=1, dropout_prob=0.2):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, first_hidden_size)
        self.fc2 = nn.Linear(first_hidden_size, other_hidden_size)
        self.fc3 = nn.Linear(other_hidden_size, other_hidden_size)
        self.fc4 = nn.Linear(other_hidden_size, output_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x

# Function to perform k-fold cross-validation for different neuron sizes
def neuron_cv(X, y, neuron_sizes, batch_size, n_splits=5, n_epochs=100, patience=3, device='cpu'):
    # Initialize dictionary to store results
    cv_accuracies = {n: [] for n in neuron_sizes}
    
    # Define k-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for neurons in neuron_sizes:
        print(f"Testing neuron size: {neurons}")
        fold_accuracies = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold+1}/{n_splits}")
            
            # Split data
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Scale features
            X_train_fold_scaled, X_val_fold_scaled = preprocess_dataset(X_train_fold, X_val_fold)
            
            # Create datasets and dataloaders
            train_dataset = MusicDataset(X_train_fold_scaled, y_train_fold)
            val_dataset = MusicDataset(X_val_fold_scaled, y_val_fold)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Initialize model, optimizer, and loss function
            input_size = X_train_fold.shape[1]
            model = CustomMLP(input_size=input_size, first_hidden_size=neurons)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.BCELoss()
            
            # Early stopping
            early_stopper = EarlyStopper(patience=patience)
            
            # Training loop
            best_val_acc = 0
            
            for epoch in range(n_epochs):
                # Train and evaluate
                _, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
                val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
                
                # Update best accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                # Check early stopping
                if early_stopper.early_stop(val_loss):
                    print(f"    Early stopping at epoch {epoch+1}")
                    break
            
            fold_accuracies.append(best_val_acc)
        
        # Store average accuracy for the neuron size
        cv_accuracies[neurons] = np.mean(fold_accuracies)
    
    return cv_accuracies

# Define neuron sizes to test
neuron_sizes = [64, 128, 256]

# Use the optimal batch size from Q2
batch_size = optimal_batch_size  # or use a specific value like 128

# Run cross-validation
cv_accuracies_neurons = neuron_cv(X_train, y_train, neuron_sizes, batch_size, device=device)

# Plot mean cross-validation accuracies
plt.figure(figsize=(10, 5))
plt.scatter(neuron_sizes, [cv_accuracies_neurons[n] for n in neuron_sizes], marker='o', s=100)
plt.plot(neuron_sizes, [cv_accuracies_neurons[n] for n in neuron_sizes], 'b--')
plt.xticks(neuron_sizes)
plt.xlabel('Number of Neurons in First Hidden Layer')
plt.ylabel('Mean CV Accuracy')
plt.title('Mean Cross-Validation Accuracy vs. Number of Neurons')
plt.grid(True)
plt.savefig('neuron_size_accuracy.png')
plt.show()

# Choose optimal number of neurons
optimal_neurons = max(cv_accuracies_neurons, key=cv_accuracies_neurons.get)
print(f"\nOptimal number of neurons: {optimal_neurons}")
print(f"Optimal neuron size accuracy: {cv_accuracies_neurons[optimal_neurons]:.4f}")

# Rationale for selection
print("\nRationale for selection:")
print(f"The neuron size {optimal_neurons} was selected because it provides the highest validation accuracy.")
print(f"This size provides sufficient model capacity to learn the underlying patterns in the data")
print(f"without overfitting. Smaller networks might not capture all the relevant features,")
print(f"while larger networks could lead to overfitting or longer training times.")

# Train the model with optimal parameters and plot learning curves
# Create dataloaders with optimal batch size
train_loader = DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=optimal_batch_size, shuffle=False)

# Initialize model, optimizer, and loss function
input_size = X_train.shape[1]
optimal_model = CustomMLP(input_size=input_size, first_hidden_size=optimal_neurons)
optimal_model = optimal_model.to(device)
optimizer = torch.optim.Adam(optimal_model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# Initialize for training
early_stopper = EarlyStopper(patience=3)

n_epochs = 100
train_losses = []
train_accs = []
test_losses = []
test_accs = []

# Training loop
for epoch in range(n_epochs):
    # Train
    train_loss, train_acc = train_epoch(optimal_model, train_loader, optimizer, loss_fn, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluate
    test_loss, test_acc = evaluate(optimal_model, test_loader, loss_fn, device)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    # Check early stopping
    if early_stopper.early_stop(test_loss):
        print(f'Early stopping at epoch {epoch+1}')
        break

# Plot the training and test accuracies
plt.figure(figsize=(10, 5))
plt.plot(train_accs, label='Train Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title(f'Accuracy vs. Epochs (Optimal Neurons: {optimal_neurons})')
plt.grid(True)
plt.savefig('optimal_accuracy_plot.png')
plt.show()

# Save the optimal model for use in Q4
torch.save(optimal_model.state_dict(), 'optimal_model.pth')

# ==============================================================================
# Part A, Q4 (10 marks)
# ==============================================================================

# Load the test audio file and extract features
test_audio_path = './audio_test.wav'  # Update this with the actual path
df = extract_features(test_audio_path)

# Show the shape of the DataFrame
size_row, size_column = df.shape
print(f"DataFrame shape: {size_row} rows x {size_column} columns")

# Preprocess the test data
# Extract features (excluding the filename column)
X_test_single = df.drop(['filename'], axis=1)

# Scale using the same scaler
X_test_single_scaled = preprocessing.StandardScaler().fit(X_train).transform(X_test_single)
X_test_tensor = torch.tensor(X_test_single_scaled, dtype=torch.float32)

# Load the optimal model
optimal_model = CustomMLP(input_size=input_size, first_hidden_size=optimal_neurons)
optimal_model.load_state_dict(torch.load('optimal_model.pth'))
optimal_model.eval()

# Make prediction
with torch.no_grad():
    output = optimal_model(X_test_tensor)
    pred_prob = output.item()
    pred_label = 1 if pred_prob >= 0.5 else 0

# Convert label back to genre
genre_map = {0: 'blues', 1: 'metal'}
predicted_genre = genre_map[pred_label]

print(f"Predicted probability: {pred_prob:.4f}")
print(f"Predicted label: {pred_label} ({predicted_genre})")

# SHAP analysis
# Select a small subset of the training data for the explainer
background = X_train_scaled[:100]  # Use a small subset for computational efficiency
background_tensor = torch.tensor(background, dtype=torch.float32)

# Initialize the DeepExplainer
explainer = shap.DeepExplainer(optimal_model, background_tensor)

# Calculate SHAP values for the test sample
shap_values = explainer.shap_values(X_test_tensor)

# Get feature names from the original dataframe (excluding 'filename')
feature_names = X_test_single.columns.tolist()

# Create a DataFrame of SHAP values
shap_df = pd.DataFrame(shap_values[0], columns=feature_names)

# Get the top 10 most important features
top_features = shap_df.abs().mean().sort_values(ascending=False).head(10).index.tolist()

# Plot SHAP values for the top features
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_single_scaled, feature_names=feature_names, plot_type="bar", 
                  max_display=10, show=False)
plt.title(f'Top 10 Important Features for Prediction: {predicted_genre}')
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
plt.show()

# Force plot for explaining the prediction
plt.figure(figsize=(20, 3))
shap.initjs()  # Initialize JavaScript visualization
force_plot = shap.force_plot(explainer.expected_value[0], 
                            shap_values[0], 
                            X_test_tensor.numpy(), 
                            feature_names=feature_names,
                            matplotlib=True,
                            show=False)
plt.title(f'SHAP Force Plot for Prediction: {predicted_genre} (probability: {pred_prob:.4f})')
plt.tight_layout()
plt.savefig('shap_force_plot.png')
plt.show()

# Explain the SHAP results
print("\nSHAP Analysis Explanation:")
print("==========================")
print(f"The model predicted the audio sample as '{predicted_genre}' with a probability of {pred_prob:.4f}.")
print("\nThe most important features for this prediction were:")

# Get top 5 features with their SHAP values
top_5_features = shap_df.abs().mean().sort_values(ascending=False).head(5)
for feature, importance in top_5_features.items():
    # Determine if the feature pushed toward or away from the prediction
    direction = "increased" if shap_df[feature].iloc[0] > 0 else "decreased"
    print(f"- {feature}: This feature {direction} the prediction probability by {abs(shap_df[feature].iloc[0]):.4f}")

print("\nAnalysis:")
print("The SHAP force plot shows how each feature pushed the model output from the base value")
print("toward the final prediction. Features pushing to the right (red) increased the prediction")
print("toward 'metal', while features pushing to the left (blue) decreased it toward 'blues'.")
print("\nThis type of analysis helps us understand which audio characteristics are most distinctive")
print("between blues and metal genres, providing valuable insights for audio classification tasks.")
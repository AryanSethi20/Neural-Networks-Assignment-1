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

# CS4001/4042 Assignment 1
# Part B Solution
# Python 3.10.9 compatible

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import from common_utils
import common_utils

# Set seed using the provided utility
common_utils.set_seed(42)

# Question B1: Neural Network for Tabular Data with PyTorch Tabular
print("Part B, Question 1: PyTorch Tabular Implementation")

# Read the dataset
df = pd.read_csv('hdb_price_prediction.csv')
print(f"Dataset shape: {df.shape}")
print(f"Year range: {df['year'].min()} to {df['year'].max()}")

# Split data by year
train_df = df[df['year'] <= 2020].copy()
test_df_2021 = df[df['year'] == 2021].copy()
test_df_2022 = df[df['year'] == 2022].copy() 
test_df_2023 = df[df['year'] == 2023].copy()

print(f"Train set (≤2020): {train_df.shape[0]} records")
print(f"Test set (2021): {test_df_2021.shape[0]} records")
print(f"Test set (2022): {test_df_2022.shape[0]} records")
print(f"Test set (2023): {test_df_2023.shape[0]} records")

# Define features as specified in the assignment
categorical_features = ['month', 'town', 'flat_model_type', 'storey_range']
continuous_features = ['dist_to_nearest_stn', 'dist_to_dhoby', 'degree_centrality', 
                      'eigenvector_centrality', 'remaining_lease_years', 'floor_area_sqm']
target = 'resale_price'

# Import PyTorch Tabular libraries
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

# Configure the model
data_config = DataConfig(
    target=[target],
    continuous_cols=continuous_features,
    categorical_cols=categorical_features,
    continuous_feature_transform=None,  # Auto-scaling will be applied
)

trainer_config = TrainerConfig(
    auto_lr_find=True,  # Automatically tune learning rate
    batch_size=1024,
    max_epochs=50,
    gpus=0,  # CPU only
    early_stopping="valid_loss",
    early_stopping_patience=3,  # Using same patience as in common_utils.EarlyStopper default
)

optimizer_config = OptimizerConfig(
    optimizer="Adam"  # No need to set learning rate (auto-tuned)
)

model_config = CategoryEmbeddingModelConfig(
    task="regression",
    layers=[50],  # 1 hidden layer with 50 neurons as required
    activation="ReLU",
    dropout=0.1,
)

# Initialize and train the model
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

print("Training the PyTorch Tabular model...")
tabular_model.fit(train=train_df, validation=test_df_2021)

# Evaluate the model on 2021 test data
pred_df = tabular_model.predict(test_df_2021)
test_preds = pred_df[target + "_prediction"].values
test_actuals = test_df_2021[target].values

# Calculate metrics
rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
r2 = r2_score(test_actuals, test_preds)

print(f"\nB1 Results on 2021 Test Set:")
print(f"RMSE: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Calculate errors for each test sample
test_df_2021['predicted'] = test_preds
test_df_2021['error'] = np.abs(test_df_2021['predicted'] - test_df_2021[target])

# Show top 25 samples with largest errors
top_25_errors = test_df_2021.sort_values(by='error', ascending=False).head(25)
print("\nTop 25 samples with largest prediction errors:")
print(top_25_errors[['year', 'month', 'town', 'flat_model_type', 'floor_area_sqm', 'resale_price', 'predicted', 'error']])

# Question B2: Neural Network with PyTorch-WideDeep
print("\nPart B, Question 2: PyTorch-WideDeep Implementation")

from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import TabMlp, WideDeep
from pytorch_widedeep import Trainer
from pytorch_widedeep.metrics import R2Score

# For B2, test set includes 2021 and after
test_df_b2 = df[df['year'] >= 2021].copy()
print(f"Train set (≤2020): {train_df.shape[0]} records")
print(f"Test set (≥2021): {test_df_b2.shape[0]} records")

# Prepare data for TabPreprocessor
X_train = train_df.copy()
y_train = X_train.pop(target)
X_test = test_df_b2.copy()
y_test = X_test.pop(target)

# Preprocess the tabular data
tab_preprocessor = TabPreprocessor(
    categorical_cols=categorical_features,
    continuous_cols=continuous_features,
    scale=True
)

X_tab_train = tab_preprocessor.fit_transform(X_train)
X_tab_test = tab_preprocessor.transform(X_test)

# Create the TabMlp model with 2 hidden layers (200, 100)
tab_mlp = TabMlp(
    column_idx=tab_preprocessor.column_idx,
    mlp_hidden_dims=[200, 100],
    mlp_activation="relu",
    embed_input=tab_preprocessor.embeddings_input,
    continuous_cols=continuous_features
)

# Combine the components
model = WideDeep(deeptabular=tab_mlp)

# Create a Trainer
trainer = Trainer(
    model,
    objective="regression",
    optimizers="Adam",
    lr=0.001,
    metrics=[R2Score],
    callbacks=None,
    verbose=1
)

# Train the model
print("Training the PyTorch-WideDeep model...")
trainer.fit(
    X_tab=X_tab_train,
    target=y_train,
    n_epochs=60,
    batch_size=64,
    val_split=0.1,
    num_workers=0
)

# Evaluate on test set
preds = trainer.predict(X_tab=X_tab_test)
rmse_b2 = np.sqrt(mean_squared_error(y_test, preds))
r2_b2 = r2_score(y_test, preds)

print(f"\nB2 Results on Test Set (≥2021):")
print(f"RMSE: ${rmse_b2:.2f}")
print(f"R² Score: {r2_b2:.4f}")

# Question B3: Model Explainability with Captum
print("\nPart B, Question 3: Model Explainability with Captum")

# Use the same train/test split as B1 but only with numeric features
X_train_numeric = train_df[continuous_features].copy()
y_train_numeric = train_df[target].copy()
X_test_numeric = test_df_2021[continuous_features].copy()
y_test_numeric = test_df_2021[target].copy()

# Standardize features - using function from common_utils would require slight modification, 
# as it expects DataFrames, but we'll keep it consistent with the rest of the solution
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_numeric.values, dtype=torch.float32).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_numeric.values, dtype=torch.float32).reshape(-1, 1)

# We could use the MLP_Custom from common_utils, but it has a different architecture
# than what's required for Question B3 (3 hidden layers with 5 neurons each)
# So we'll define a simple neural network specifically for this task
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Create and train the model
model_explain = SimpleNN(len(continuous_features))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model_explain.parameters(), lr=0.001)

# Train the model using EarlyStopper from common_utils
n_epochs = 100
batch_size = 64
n_samples = X_train_tensor.shape[0]
n_batches = n_samples // batch_size

# Initialize early stopper
early_stopper = common_utils.EarlyStopper(patience=3, min_delta=0)

print("Training the SimpleNN model for explainability...")
for epoch in range(n_epochs):
    model_explain.train()
    for i in range(0, n_samples, batch_size):
        X_batch = X_train_tensor[i:i+batch_size]
        y_batch = y_train_tensor[i:i+batch_size]
        
        optimizer.zero_grad()
        y_pred = model_explain(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    
    # Calculate validation loss for early stopping
    model_explain.eval()
    with torch.no_grad():
        y_pred = model_explain(X_test_tensor)
        test_loss = loss_fn(y_pred, y_test_tensor).item()
        
        # Check for early stopping
        if early_stopper.early_stop(test_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print progress every 10 epochs
        if (epoch+1) % 10 == 0:
            y_pred_train = model_explain(X_train_tensor)
            train_loss = loss_fn(y_pred_train, y_train_tensor).item()
            print(f"Epoch {epoch+1}: Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}")

# Evaluate the model
model_explain.eval()
with torch.no_grad():
    y_pred = model_explain(X_test_tensor)
    test_mse = loss_fn(y_pred, y_test_tensor).item()
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test_tensor.numpy(), y_pred.numpy())
    
print(f"\nSimpleNN Model Performance:")
print(f"RMSE: ${test_rmse:.2f}")
print(f"R² Score: {test_r2:.4f}")

# Import Captum for model explainability
from captum.attr import Saliency, InputXGradient, IntegratedGradients, DeepLift, GradientShap, FeatureAblation

# Use only first 1000 test samples to reduce computation time
n_explain_samples = min(1000, X_test_tensor.shape[0])
X_subset = X_test_tensor[:n_explain_samples]

# Initialize attribution methods
saliency = Saliency(model_explain)
input_x_gradient = InputXGradient(model_explain)
integrated_gradients = IntegratedGradients(model_explain)
deep_lift = DeepLift(model_explain)
gradient_shap = GradientShap(model_explain)
feature_ablation = FeatureAblation(model_explain)

# Compute attributions
attributions = {}
print("\nComputing feature attributions with different methods...")

# Input x Gradient
attributions['Input x Gradient'] = input_x_gradient.attribute(X_subset).detach().numpy()

# Integrated Gradients
attributions['Integrated Gradients'] = integrated_gradients.attribute(X_subset).detach().numpy()

# DeepLift
attributions['DeepLift'] = deep_lift.attribute(X_subset).detach().numpy()

# GradientSHAP
baseline = torch.zeros((1, X_subset.shape[1]))
attributions['GradientSHAP'] = gradient_shap.attribute(X_subset, baselines=baseline).detach().numpy()

# Feature Ablation
attributions['Feature Ablation'] = feature_ablation.attribute(X_subset).detach().numpy()

# Compute mean attributions across samples for each feature
mean_attributions = {
    method: np.abs(attr).mean(axis=0) for method, attr in attributions.items()
}

# Plot feature attributions
plt.figure(figsize=(15, 8))
x = range(len(continuous_features))
width = 0.15
offset = 0

for i, (method, attr) in enumerate(mean_attributions.items()):
    plt.bar([p + offset for p in x], attr, width, label=method)
    offset += width

plt.xlabel('Features')
plt.ylabel('Mean Attribution Score')
plt.title('Feature Importance by Explainability Method')
plt.xticks([i + 0.3 for i in x], continuous_features, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('feature_attributions.png')

# Identify top three features based on average of all methods
avg_attributions = np.mean([attr for attr in mean_attributions.values()], axis=0)
top_features_idx = np.argsort(avg_attributions)[::-1][:3]
top_features = [continuous_features[i] for i in top_features_idx]

print(f"\nTop three most important features:")
for i, feature in enumerate(top_features):
    print(f"{i+1}. {feature}: {avg_attributions[top_features_idx[i]]:.4f}")

# Question B4: Model Degradation Analysis
print("\nPart B, Question 4: Model Degradation Analysis")

# Evaluate the B1 model on 2022 and 2023 data
pred_df_2022 = tabular_model.predict(test_df_2022)
test_preds_2022 = pred_df_2022[target + "_prediction"].values
test_actuals_2022 = test_df_2022[target].values
rmse_2022 = np.sqrt(mean_squared_error(test_actuals_2022, test_preds_2022))
r2_2022 = r2_score(test_actuals_2022, test_preds_2022)

pred_df_2023 = tabular_model.predict(test_df_2023)
test_preds_2023 = pred_df_2023[target + "_prediction"].values
test_actuals_2023 = test_df_2023[target].values
rmse_2023 = np.sqrt(mean_squared_error(test_actuals_2023, test_preds_2023))
r2_2023 = r2_score(test_actuals_2023, test_preds_2023)

print(f"Model Performance Across Years:")
print(f"2021 - RMSE: ${rmse:.2f}, R²: {r2:.4f}")
print(f"2022 - RMSE: ${rmse_2022:.2f}, R²: {r2_2022:.4f}")
print(f"2023 - RMSE: ${rmse_2023:.2f}, R²: {r2_2023:.4f}")

# Check for data drift using Alibi Detect
from alibi_detect.cd import TabularDrift

# Sample 1000 records from train and 2023 test datasets
np.random.seed(SEED)
train_sample = train_df.sample(1000, random_state=SEED)
test_sample_2023 = test_df_2023.sample(min(1000, test_df_2023.shape[0]), random_state=SEED)

# Convert categorical features to numerical using one-hot encoding
X_train_drift = pd.get_dummies(train_sample[continuous_features + categorical_features])
X_test_drift = pd.get_dummies(test_sample_2023[continuous_features + categorical_features])

# Align columns (in case some values only appear in test or train)
common_columns = X_train_drift.columns.intersection(X_test_drift.columns)
X_train_drift = X_train_drift[common_columns]
X_test_drift = X_test_drift[common_columns]

print(f"\nChecking for data drift between train (≤2020) and 2023 data...")
# Initialize drift detector
drift_detector = TabularDrift(
    X_train_drift.values,
    p_val=0.05,
    categories_per_feature=None,
    preprocess_fn=None,
    feature_names=list(X_train_drift.columns)
)

# Predict if drift occurred
drift_preds = drift_detector.predict(X_test_drift.values)
is_drift = drift_preds['data']['is_drift']
p_vals = drift_preds['data']['p_val']
feature_scores = dict(zip(X_train_drift.columns, p_vals))

print(f"Overall drift detected: {is_drift}")
print("\nFeatures with significant drift (p-value < 0.05):")
drifted_features = {feat: p_val for feat, p_val in feature_scores.items() if p_val < 0.05}
for feat, p_val in sorted(drifted_features.items(), key=lambda x: x[1]):
    print(f"{feat}: p-value = {p_val:.4f}")

# Addressing model degradation by fine-tuning
print("\nAddressing model degradation by fine-tuning the model on recent data...")

# Combine 2021 and 2022 data for fine-tuning
fine_tune_df = pd.concat([test_df_2021, test_df_2022])

# Fine-tune the model
tabular_model.fit(train=fine_tune_df)

# Evaluate fine-tuned model on 2023 data
pred_df_2023_ft = tabular_model.predict(test_df_2023)
test_preds_2023_ft = pred_df_2023_ft[target + "_prediction"].values
rmse_2023_ft = np.sqrt(mean_squared_error(test_actuals_2023, test_preds_2023_ft))
r2_2023_ft = r2_score(test_actuals_2023, test_preds_2023_ft)

print(f"\nFine-tuned Model Performance on 2023 data:")
print(f"Before fine-tuning - RMSE: ${rmse_2023:.2f}, R²: {r2_2023:.4f}")
print(f"After fine-tuning - RMSE: ${rmse_2023_ft:.2f}, R²: {r2_2023_ft:.4f}")
print(f"Improvement in R²: {r2_2023_ft - r2_2023:.4f}")

print("\nAssignment Part B Complete")
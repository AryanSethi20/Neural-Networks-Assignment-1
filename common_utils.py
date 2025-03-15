import tqdm
import time
import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import os

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from scipy.io import wavfile as wav

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def split_dataset(df, columns_to_drop, test_size, random_state):
    
    label_encoder = preprocessing.LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)

    df_train2 = df_train.drop(columns_to_drop,axis=1)
    y_train2 = df_train['label'].to_numpy()

    df_test2 = df_test.drop(columns_to_drop,axis=1)
    y_test2 = df_test['label'].to_numpy() 

    return df_train2, y_train2, df_test2, y_test2

def preprocess_dataset(df_train, df_test):

    standard_scaler = preprocessing.StandardScaler()
    df_train_scaled = standard_scaler.fit_transform(df_train)

    df_test_scaled = standard_scaler.transform(df_test)

    return df_train_scaled, df_test_scaled

def set_seed(seed = 0):
    '''
    set random seed
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def extract_features(filepath):
    
    '''
    This function reads the content in a directory and for each audio file detected
    reads the file and extracts relevant features using librosa library for audio
    signal processing
    '''
    # Reading audio file
    try:
        y, sr = librosa.load(filepath, mono=True)
        
        # if y.ndim > 1:
        #     y = np.mean(y, axis=0)
    except Exception as e:
        print(f"Error loading audio: {e}")
        raise
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony, perceptr = librosa.effects.harmonic(y), librosa.effects.percussive(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    frames_to_time = librosa.frames_to_time(onset_frames[:20], sr=sr)

    features = [
                f"{filepath}",
                np.mean(chroma_stft), np.var(chroma_stft),
                np.mean(rms), np.var(rms),
                np.mean(spec_cent), np.var(spec_cent),
                np.mean(spec_bw), np.var(spec_bw),
                np.mean(rolloff), np.var(rolloff),
                np.mean(zcr), np.var(zcr),
                np.mean(harmony), np.var(harmony),
                np.mean(perceptr), np.var(perceptr),
                float(tempo)
            ]

    for coeff in mfcc:
        features.append(np.mean(coeff))
        features.append(np.var(coeff))

    columns=['filename',
         'chroma_stft_mean', 'chroma_stft_var',
         'rms_mean', 'rms_var',
         'spectral_centroid_mean', 'spectral_centroid_var',
         'spectral_bandwidth_mean', 'spectral_bandwidth_var',
         'rolloff_mean', 'rolloff_var',
         'zero_crossing_rate_mean','zero_crossing_rate_var',
         'harmony_mean', 'harmony_var',
         'perceptr_mean', 'perceptr_var',
         'tempo'] + \
         [f'mfcc{i+1}_{stat}' for i in range(20) for stat in ['mean', 'var']]

    feature_set = pd.DataFrame([features], columns=columns)
         
    return feature_set

def generate_folds(n_folds, parameters, training_features, training_labels):
    X_train_scaled_dict = {param: [] for param in parameters}
    X_val_scaled_dict = {param: [] for param in parameters}
    y_train_dict = {param: [] for param in parameters}
    y_val_dict = {param: [] for param in parameters}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for param in parameters:
        for train_idx, val_idx in kf.split(training_features):
            X_train_fold, X_val_fold = training_features.iloc[train_idx], training_features.iloc[val_idx]
            y_train_fold, y_val_fold = training_labels[train_idx], training_labels[val_idx]
            
            X_train_fold_scaled, X_val_fold_scaled = preprocess_dataset(X_train_fold, X_val_fold)
            
            X_train_scaled_dict[param].append(X_train_fold_scaled)
            X_val_scaled_dict[param].append(X_val_fold_scaled)
            y_train_dict[param].append(y_train_fold)
            y_val_dict[param].append(y_val_fold)
            
    return X_train_scaled_dict, X_val_scaled_dict, y_train_dict, y_val_dict


# early stopping obtained from tutorial
class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class MLP_Custom(nn.Module):
    def __init__(self, input_size, first_hidden_size, other_hidden_size=128, output_size=1, dropout_prob=0.2):
        super(MLP_Custom, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, first_hidden_size)
        self.fc2 = torch.nn.Linear(first_hidden_size, other_hidden_size)
        self.fc3 = torch.nn.Linear(other_hidden_size, other_hidden_size)
        self.fc4 = torch.nn.Linear(other_hidden_size, output_size)
        
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc4(x))
        return x

class MLP_hidden(nn.Module):

    def __init__(self, no_features, no_neuron, no_hidden = 128, no_labels = 1):
        super().__init__()
        self.mlp_stack = nn.Sequential(
            # YOUR CODE HERE
            nn.Linear(no_features, no_neuron),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(no_neuron, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(no_hidden, no_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(no_hidden, no_labels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        logits = self.mlp_stack(x)
        return logits
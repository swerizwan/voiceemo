import re
import random
import math
import numpy as np
import argparse
from tqdm import tqdm
import os
import shutil
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import get_dataloaders
from Facesynthesizer import VERHM, RavVERHM, MeadVERHM, VocLAE
import librosa
from sklearn.model_selection import train_test_split


# Set random seed for reproducibility
# This ensures that any random operations are repeatable and produce the same results every time the code is run.
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Define constants
# max_pad_len determines the fixed length to which MFCCs (Mel-Frequency Cepstral Coefficients) will be padded.
max_pad_len = 100

# Function to pad MFCCs to a fixed length
# This function takes an MFCC array and pads or truncates it to ensure a consistent input size for the model.
def pad_mfcc(mfcc, max_pad_len):
    if mfcc.shape[1] > max_pad_len:
        # If the number of columns in mfcc is greater than max_pad_len, truncate it to max_pad_len
        mfcc = mfcc[:, :max_pad_len]
    else:
        # If the number of columns in mfcc is less than max_pad_len, pad the array with zeros to reach max_pad_len
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

# Custom dataset class for MEAD
# This class inherits from PyTorch's Dataset class and is used to load and process data for training and evaluation.
class MeadDataset(Dataset):
    def __init__(self, file_paths, labels):
        # Initialization method to set up file paths and labels
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        # Returns the number of samples in the dataset
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Loads an audio file and its corresponding label at the given index
        y, sr = librosa.load(self.file_paths[idx], sr=None)  # Load the audio file with its sampling rate
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Compute MFCCs with 13 coefficients
        mfccs = pad_mfcc(mfccs, max_pad_len)  # Pad or truncate MFCCs to max_pad_len
        # Return MFCCs as a tensor of type float32 and the label as a tensor of type long
        return torch.tensor(mfccs, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


# Function to extract features from audio files
# This function takes a list of file paths and extracts MFCC features for each audio file,
# padding them to ensure a consistent input length.
def extract_features(file_paths, max_pad_len):
    features = []
    for path in file_paths:
        # Load the audio file with its sampling rate
        y, sr = librosa.load(path, sr=None)
        # Compute MFCCs with 13 coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Pad or truncate MFCCs to max_pad_len
        mfccs = pad_mfcc(mfccs, max_pad_len)
        # Append the processed MFCCs to the features list
        features.append(mfccs)
    # Return the features as a NumPy array
    return np.array(features)

# Function to train on MEAD dataset
# This function handles the preparation of data, the training of the model, and evaluation on validation and test sets.
def train_mead(args):
    data_dir = "mead/"
    file_paths = []

    # Traverse the dataset directory to gather all audio file paths
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_paths.append(os.path.join(root, file))

    # Shuffle the file paths to ensure random distribution of data
    random.shuffle(file_paths)
    total_samples = len(file_paths)
    
    # Split the data into training (70%), validation (15%), and test (15%) sets
    train_size = int(0.7 * total_samples)
    val_size = test_size = (total_samples - train_size) // 2
    
    train_files = file_paths[:train_size]
    val_files = file_paths[train_size:train_size + val_size]
    test_files = file_paths[train_size + val_size:]

    # Extract features for each set
    X_train = extract_features(train_files, max_pad_len)
    X_val = extract_features(val_files, max_pad_len)
    X_test = extract_features(test_files, max_pad_len)

    # Generate random labels for each set (assuming a classification problem with 7 classes)
    y_train = np.random.randint(0, 7, size=len(train_files))
    y_val = np.random.randint(0, 7, size=len(val_files))
    y_test = np.random.randint(0, 7, size=len(test_files))

    # Create Dataset objects for each set
    train_dataset = MeadDataset(train_files, y_train)
    val_dataset = MeadDataset(val_files, y_val)
    test_dataset = MeadDataset(test_files, y_test)

    # Create DataLoader objects for each set
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Instantiate the model, loss function, optimizer, and learning rate scheduler
    model = MeadVERHM()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Training loop
    for epoch in range(args.epochsmead):
        model.train()
        train_loss = 0.0

        # Iterate over the training data
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            train_loss += loss.item() * inputs.size(0)  # Accumulate training loss

        train_loss = train_loss / len(train_loader.dataset)  # Average training loss

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # Iterate over the validation data
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get predictions
                total += labels.size(0)  # Total number of samples
                correct += (predicted == labels).sum().item()  # Correct predictions count
                val_loss += criterion(outputs, labels).item()  # Accumulate validation loss

        val_loss = val_loss / len(val_loader.dataset)  # Average validation loss
        val_accuracy = correct / total  # Validation accuracy

        # print(f"Epoch {epoch+1}/{args.epochsmead}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

    # Testing phase
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    predicted_timestamps = []
    ground_truth_timestamps = []

    # Iterate over the test data
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get predictions
            total_test += labels.size(0)  # Total number of samples
            correct_test += (predicted == labels).sum().item()  # Correct predictions count
            test_loss += criterion(outputs, labels).item()  # Accumulate test loss
            predicted_timestamps.extend(predicted.tolist())  # Collect predictions
            ground_truth_timestamps.extend(labels.tolist())  # Collect ground truth labels

    test_loss = test_loss / len(test_loader.dataset)  # Average test loss
    test_accuracy = correct_test / total_test  # Test accuracy

    # print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Calculate Lip Average Error (LAE)
    MeadLAE = MeadVERHM()
    LAE = MeadLAE.MeadLAE(predicted_timestamps, ground_truth_timestamps)
    print(f"Lip Average Error (LAE): {LAE:.4f}")


# Define constants
# Mapping of emotion codes to their corresponding emotion labels
emotions = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}
# Maximum padding length for MFCCs
max_pad_len = 400

# Define functions

# Function to pad MFCCs to a fixed length
def pad_mfccrav(mfcc):
    if mfcc.shape[1] > max_pad_len:
        # If the number of columns in mfcc is greater than max_pad_len, truncate it to max_pad_len
        mfcc = mfcc[:, :max_pad_len]
    else:
        # If the number of columns in mfcc is less than max_pad_len, pad the array with zeros to reach max_pad_len
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

# Function to pad MFCC to a defined length
def pad_mfccrav(mfcc, max_length=100):
    if mfcc.shape[1] > max_length:
        return mfcc[:, :max_length]
    else:
        pad_width = max_length - mfcc.shape[1]
        return np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')

# Dictionary of emotions
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# Function to train on the RAVDESS dataset
def train_ravdess(args):
    # Initialize lists to store data, labels, and synthetic vertices
    data = []
    labels = []
    synthetic_vertices = []

    # Iterate over actor directories in the dataset path
    for actor_dir in tqdm(os.listdir(args.dataset_path), desc="Processing audio files"):
        actor_path = os.path.join(args.dataset_path, actor_dir)
        for audio_file in os.listdir(actor_path):
            # Check if the file format is valid
            if len(audio_file.split("-")) < 3:
                print(f"Unexpected file format: {audio_file}. Skipping this file.")
                continue

            # Extract emotion code from the file name and get the corresponding emotion label
            emotion_key = audio_file.split("-")[2]
            emotion = emotions[emotion_key]
            
            # Load the audio file with a sampling rate of 44100 Hz
            full_path = os.path.join(actor_path, audio_file)
            y, sr = librosa.load(full_path, sr=44100)
            # Compute MFCCs from the audio signal
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            # Pad the MFCC to the defined length
            mfcc = pad_mfccrav(mfcc)
            
            # Append the processed MFCC and corresponding label to their respective lists
            data.append(mfcc)
            labels.append(list(emotions.values()).index(emotion))
            
            # Generate synthetic vertices for demonstration
            synthetic_vertex = [random.uniform(0, 1), random.uniform(0, 1)]
            synthetic_vertices.append(synthetic_vertex)

    # Convert data, labels, and synthetic vertices to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)
    synthetic_vertices = np.array(synthetic_vertices)

    # Convert NumPy arrays to PyTorch tensors
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    synthetic_vertices = torch.tensor(synthetic_vertices, dtype=torch.float32)

    # Calculate dataset sizes for training, validation, and test sets
    total_size = len(data)
    train_size = int(0.8 * total_size)
    val_size = test_size = (total_size - train_size) // 2

    # Split the dataset into training, validation, and test sets
    dataset = TensorDataset(data, labels, synthetic_vertices)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Define dataloaders for training, validation, and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model and set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_model = RavVERHM().to(device)

    # Define loss functions for emotion classification and vertex prediction
    emotion_criterion = nn.CrossEntropyLoss()
    vertex_criterion = nn.MSELoss()
    # Define the optimizer and learning rate scheduler
    optimizer = optim.Adam(nn_model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Training loop
    print("\nTraining Model...")
    num_epochs = args.epochs
    for epoch in range(num_epochs):
        nn_model.train()
        running_loss = 0.0
        
        # Iterate over the training data
        for inputs, labels, vertices in train_loader:
            inputs, labels, vertices = inputs.to(device), labels.to(device), vertices.to(device)
            optimizer.zero_grad()  # Zero the gradients
            emotion_outputs, vertex_outputs = nn_model(inputs)  # Forward pass
            emotion_loss = emotion_criterion(emotion_outputs, labels)  # Compute emotion classification loss
            vertex_loss = vertex_criterion(vertex_outputs, vertices)  # Compute vertex prediction loss
            loss = emotion_loss + vertex_loss  # Total loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            running_loss += loss.item() * inputs.size(0)  # Accumulate training loss
        
        epoch_loss = running_loss / len(train_loader.dataset)  # Average training loss
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")
        
        # Update learning rate based on the average training loss
        scheduler.step(epoch_loss)
    
    # Final evaluation on the test set
    nn_model.eval()
    correct = 0
    total = 0
    predicted_values = []
    ground_truth_values = []
    predicted_vertices = []
    ground_truth_vertices = []

    # Iterate over the test data
    with torch.no_grad():
        for inputs, labels, vertices in test_loader:
            inputs, labels, vertices = inputs.to(device), labels.to(device), vertices.to(device)
            emotion_outputs, vertex_outputs = nn_model(inputs)  # Forward pass
            _, predicted = torch.max(emotion_outputs.data, 1)  # Get predicted emotions
            predicted_values.extend(predicted.cpu().numpy())  # Collect predicted values
            ground_truth_values.extend(labels.cpu().numpy())  # Collect ground truth values
            predicted_vertices.extend(vertex_outputs.cpu().numpy())  # Collect predicted vertices
            ground_truth_vertices.extend(vertices.cpu().numpy())  # Collect ground truth vertices

    # Convert lists to NumPy arrays
    predicted_values = np.array(predicted_values)
    ground_truth_values = np.array(ground_truth_values)
    predicted_vertices = np.array(predicted_vertices)
    ground_truth_vertices = np.array(ground_truth_vertices)

    # Calculate metrics (assuming these methods are defined in the model class)
    RavLAE = RavVERHM()
    RavLVE = RavVERHM()
    RavEVE = RavVERHM()
    LAE = RavLAE.RavLAE(predicted_values, ground_truth_values)  # Calculate Lip Average Error (LAE)
    LVE = RavLVE.RavLVE(predicted_vertices, ground_truth_vertices)  # Calculate Lip Vertex Error (LVE)
    EVE = RavEVE.RavEVE(predicted_values, ground_truth_values)  # Calculate Emotional Vertex Error (EVE)

    # Print the calculated metrics
    print(f"Lip Average Error (LAE): {LAE:.3f}")
    print(f"Lip Vertex Error (LVE): {LVE:.3f}")
    print(f"Emotional Vertex Error (EVE): {EVE:.3f}")



# Training function
def trainer(args, train_loader, dev_loader, model, optimizer, criterion, epoch=100):
    # Create a save path for model checkpoints
    save_path = os.path.join(args.dataset, args.save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)  # Remove existing directory if it exists
    os.makedirs(save_path)  # Create a new directory

    # List of subjects used for training
    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    iteration = 0  # Initialize iteration counter

    # Loop over epochs
    for e in range(epoch + 1):
        loss_log = []  # List to store training loss for each batch
        # Lists to store predicted and ground truth timestamps for each epoch
        predicted_timestamps = []
        ground_truth_timestamps = []

        # Set model to training mode
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        optimizer.zero_grad()  # Initialize optimizer gradient to zero

        # Loop over batches
        for i, (audio, vertice, template, one_hot, file_name) in pbar:
            iteration += 1  # Increment iteration counter
            # Move data to GPU
            audio, vertice, template, one_hot = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            # Forward pass and compute loss
            loss = model(audio, template, vertice, one_hot, criterion, teacher_forcing=False)
            loss.backward()  # Backward pass
            loss_log.append(loss.item())  # Store loss value

            # Store predicted and ground truth timestamps for LAE calculation
            with torch.no_grad():
                predict_timestampLAE = model.predict_timestampLAE(audio, template, one_hot)  # Assuming model has predict_timestamp method
                ground_truth_timestampLAE = model.ground_truth_timestampLAE(audio, template, one_hot)  # Assuming this function is defined

            # Update model weights and reset optimizer gradients
            if i % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Calculate Lip Average Error (LAE) for the current batch
            final_LAE = VocLAE(predict_timestampLAE, ground_truth_timestampLAE)
            pbar.set_description("(Epoch {})".format((e + 1)))

        # Calculate Lip Average Error (LAE) for the epoch
        final_LAE = VocLAE(predict_timestampLAE, ground_truth_timestampLAE)

        # Validation phase
        valid_loss_log = []  # List to store validation loss for each batch
        model.eval()  # Set model to evaluation mode

        # Loop over validation batches
        for audio, vertice, template, one_hot_all, file_name in dev_loader:
            # Move data to GPU
            audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
            train_subject = "_".join(file_name[0].split("_")[:-1])
            if train_subject in train_subjects_list:
                condition_subject = train_subject
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:, iter, :]
                loss = model(audio, template, vertice, one_hot, criterion)
                valid_loss_log.append(loss.item())
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:, iter, :]
                    loss = model(audio, template, vertice, one_hot, criterion)
                    valid_loss_log.append(loss.item())

        # Calculate mean validation loss
        current_loss = np.mean(valid_loss_log)

        # Save model checkpoint every 25 epochs or at the last epoch
        if (e > 0 and e % 25 == 0) or e == args.max_epoch:
            torch.save(model.state_dict(), os.path.join(save_path, '{}_model.pth'.format(e)))
        print("epoch: {}".format(e + 1))

    # Print the final Lip Average Error (LAE)
    print("Lip Average Error (LAE):", final_LAE)
    return model

# Testing function
@torch.no_grad()
def test(args, model, test_loader, epoch):
    # Create a result path for saving predictions
    result_path = os.path.join(args.dataset, args.result_path)
    if os.path.exists(result_path):
        shutil.rmtree(result_path)  # Remove existing directory if it exists
    os.makedirs(result_path)  # Create a new directory

    # Load model checkpoint
    save_path = os.path.join(args.dataset, args.save_path)
    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    model.load_state_dict(torch.load(os.path.join(save_path, '{}_model.pth'.format(epoch))))
    model = model.to(torch.device("cuda"))
    model.eval()  # Set model to evaluation mode

    # Loop over test batches
    for audio, vertice, template, one_hot_all, file_name in test_loader:
        # Move data to GPU
        audio, vertice, template, one_hot_all = audio.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda")
        train_subject = "_".join(file_name[0].split("_")[:-1])
        if train_subject in train_subjects_list:
            condition_subject = train_subject
            iter = train_subjects_list.index(condition_subject)
            one_hot = one_hot_all[:, iter, :]
            prediction = model.predict(audio, template, one_hot)
            prediction = prediction.squeeze()  # (seq_len, V*3)
            np.save(os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"), prediction.detach().cpu().numpy())
        else:
            for iter in range(one_hot_all.shape[-1]):
                condition_subject = train_subjects_list[iter]
                one_hot = one_hot_all[:, iter, :]
                prediction = model.predict(audio, template, one_hot)
                prediction = prediction.squeeze()  # (seq_len, V*3)
                np.save(os.path.join(result_path, file_name[0].split(".")[0] + "_condition_" + condition_subject + ".npy"), prediction.detach().cpu().numpy())

# Function to count the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Voice-Driven 3D Facial Emotion Recognition For Mental Health Monitoring')
    
    # Add argument definitions
    parser.add_argument("--lr", type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument("--dataset", type=str, default="vocaset", help='Dataset to use (vocaset, ravdess, or mead)')
    parser.add_argument("--dataset_path", type=str, default="ravdess/", help='Path to the dataset')
    parser.add_argument("--epochs", type=int, default=100, help='Number of epochs for training on RAVDESS dataset')
    parser.add_argument("--epochsmead", type=int, default=100, help='Number of epochs for training on Mead dataset')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='Number of vertices (5023*3 for vocaset)')
    parser.add_argument("--feature_dim", type=int, default=64, help='Feature dimension (64 for vocaset)')
    parser.add_argument("--period", type=int, default=30, help='Period in PPE (30 for vocaset)')
    parser.add_argument("--wav_path", type=str, default="wav", help='Path to the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='Path to the ground truth vertices')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='Number of gradient accumulation steps')
    parser.add_argument("--max_epoch", type=int, default=100, help='Maximum number of epochs')
    parser.add_argument("--device", type=str, default="cuda", help='Device to use for computation (default: cuda)')
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='Path to the personalized templates file')
    parser.add_argument("--save_path", type=str, default="save", help='Path to save the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='Path to save the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA", help='Subjects to use for training')
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA", help='Subjects to use for validation')
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA", help='Subjects to use for testing')

    # Parse the arguments
    args = parser.parse_args()

    # Choose the appropriate dataset and training function
    if args.dataset == "ravdess":
        train_ravdess(args)
    elif args.dataset == "mead":
        train_mead(args)
    elif args.dataset == "vocaset":
        # Build the model
        model = VERHM(args)
        print("Model parameters: ", count_parameters(model))

        # Ensure CUDA is available and move the model to GPU
        assert torch.cuda.is_available()
        model = model.to(torch.device("cuda"))
        
        # Load the dataset
        dataset = get_dataloaders(args)
        
        # Define the loss function
        criterion = nn.MSELoss()

        # Define the optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        
        # Train the model
        model = trainer(args, dataset["train"], dataset["valid"], model, optimizer, criterion, epoch=args.max_epoch)
        
        # Test the model
        test(args, model, dataset["test"], epoch=args.max_epoch)
    
if __name__=="__main__":
    main()

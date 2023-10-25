import os
import csv
import pandas as pd
from tqdm import tqdm
import time
import re
import numpy as np
import h5py

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from keras.models import Model



from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# Directory where to search for .h5 files
directory_path = 'DiskUnit:/path/to/Data/Folders'

# Define the number of files (boards)
boards = 20  # You can change this value as needed

# Empty Sequences Dataset fo fulfill with incoming data
working_dataset = []

# Create a list to store the extracted lists from each file
indexes_arrays = []
sequences_arrays = []

# Lists to store metrics along the training
train_accuracy = []
val_accuracy = []
train_loss = []
val_loss = []

# This function searches for h5 files of the format required
def find_and_sort_h5_files(directory):
    # Initialize lists to store file names and i values
    file_names = []
    i_values = []

    # Define the expected file pattern
    pattern = r'board_(\d+)_sequences\.h5'

    # Iterate over files in the directory
    for file_name in os.listdir(directory):
        # Check if the file matches the pattern
        match = re.match(pattern, file_name)
        if match:
            # Add the file name and i value to the lists
            file_names.append(file_name)

    # Sort file names based on the i values
    sorted_file_names = sorted(file_names, key=lambda x: int(re.search(pattern, x).group(1)))
    i_values = [int(re.search(pattern, x).group(1)) for x in sorted_file_names]
    # Return the sorted file names and the list of i values
    return sorted_file_names, i_values

# Directory where to search for .h5 files
sorted_files, board_values = find_and_sort_h5_files(directory_path)

# Print the sorted file names and i values
print("Sorted Files:")
for file_name, board_value in zip(sorted_files, board_values):
    print(f"File: {file_name}, Board: {board_value}")

# Iterate over each file
for i in range(1, boards+1):
    file_name = f'board_{i}_sequences.h5'  # File name
    file_lists = {}  # Dictionary to store the lists from each file
    
    # Open the HDF5 file in read mode
    with h5py.File(file_name, 'r') as file:
        # Access the 'indexes' and 'sequences' datasets (adjust the structure according to your files)
        print(f"Processing: {file_name}")
        indexes = file['indexes'][:]
        sequences = file['sequences'][:]
        indexed_sequences = [[lst, val] for lst, val in zip(sequences, indexes)] # Join each sequence with its label 
        
        working_dataset += indexed_sequences # Add indexed sequences for further set preparation
        # print(f"Dimensiones de indexes: {len(indexes)}")
        # print(f"Dimensiones de sequences: {len(indexes)}")  



# Once working_dataset has been created, must be shuffled
working_dataset = np.array(working_dataset, dtype=object)

# Get the number of rows (sublists) in the array.
num_rows = working_dataset.shape[0]
# Generate a random index to shuffle the rows.
random_index = np.arange(num_rows)
np.random.shuffle(random_index)
# Shuffle the array according to the random index.
working_dataset = working_dataset[random_index]
# Separate Input and Label Columns
X = working_dataset[:, 0]  # INPUTS
Y = working_dataset[:, 1]  # LABELS
# Check X and Y are separated correctly
print(f" X Dimensions : {X.shape}")
print("X dtype:", X.dtype)
# print(f" X's First Element : {X[0]}")
print(f" X[0]'s First Element dimensions : {X[0].shape}")
print("X[0] dtype:", X.dtype)
print(f" Y Dimensions : {Y.shape}")
# print(f" Y's First Element : {Y[0]}")
# Empty working_dataset to free memory
working_dataset = []

print(f"\n\nNumpy Array Info\n")

# Convert Numpy Array to a TensorFlow tensor
X_tensors = [tf.constant(sequence, dtype=tf.float32) for sequence in X]
X_tensor = tf.stack(X_tensors)
Y_tensor = tf.convert_to_tensor(Y, dtype=tf.int32)
##########################################################################
print(f"X_tensor shape : {X_tensor.shape}")
print(f"X_tensor dtype : {X_tensor.dtype}")

# print(f"X_tensor[0] value : {X_tensor[0]}")
print(f"X_tensor[0] dtype : {X_tensor[0].dtype}")
print(f"X_tensor[0] shape : {X_tensor[0].shape}")
##########################################################################
print(f"\n Tensors Info\n")

print(f"Y_tensor shape : {Y_tensor.shape}")
print(f"Y_tensor dtype : {Y_tensor.dtype}")

print(f"Y_tensor values : {Y_tensor[0:5]}")
print(f"Y_tensor[0] shape : {Y_tensor[0].dtype}")
print(f"Y_tensor[0] dtype : {Y_tensor[0].shape}")

# print(f"\n\nY values : {Y[0:4]}\n\n")
# print(f" X Dimensions : {X.shape}")
# print(f" X's First Element : {X[0]}")

# divide total samples between training data and temp data
num_samples_total = X_tensor.shape[0]
split_index_train = int(0.7 * num_samples_total) # specify training ratio
x_train = X_tensor[:split_index_train]
x_temp = X_tensor[split_index_train:]
y_train = Y_tensor[:split_index_train]
y_temp = Y_tensor[split_index_train:]

# divide temp data to compose validation and test data 
num_samples_testval = x_temp.shape[0]
split_index_test = int(0.5 * num_samples_testval) # specify validation ratio from temp data
x_val = x_temp[:split_index_test]
x_test = x_temp[split_index_test:]
y_val = y_temp[:split_index_test]
y_test = y_temp[split_index_test:]

# Split the data into training and testing sets in a 70-30 proportion. ALTERNATIVE NOT WORKING
# x_train, x_temp, y_train, y_temp = train_test_split(X_tensor, Y_tensor, test_size = 0.3, shuffle = False) # No se porque no funciona :_(
# x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size = 0.5, shuffle = False)

print("\n\ Training Set Info \n\n")
print(f"x_train shape : {x_train.shape}")
print(f"x_train dtype : {x_train.dtype}")
# print(f"x_train values[0:3] :{x_train[0:3]}")
print(f"\ny_train shape : {y_train.shape}")
print(f"y_train dtype : {y_train.dtype}")
# print(f"y_train values[0:30] :{y_train[0:30]}")
print("\n\ Validation Set Info \n\n")
print(f"x_val shape : {x_val.shape}")
print(f"x_val dtype : {x_val.dtype}")
# print(f"x_val values[0:3] :{x_val[0:3]}")
print(f"\ny_val shape : {y_val.shape}")
print(f"y_val dtype : {y_val.dtype}")
# print(f"y_val values[0:30] :{y_val[0:30]}")
print("\n\ Test Set Info \n\n")
print(f"x_test shape : {x_test.shape}")
print(f"x_test dtype : {x_test.dtype}")
# print(f"x_test values[0:3] :{x_test[0:3]}")
print(f"\ny_test shape : {y_test.shape}")
print(f"y_test dtype : {y_test.dtype}")
# print(f"y_test values[0:30] :{y_test[0:30]}")

#############################################################
#                                                           #
#                  BUILDING 1D CNN MODEL                    #
#                                                           #
#############################################################


# Define the input size (sequence length) and the number of channels (features)
sequence_length = X[0].shape[0] # number of values per sequence
print(f"Shape: {sequence_length}")
number_of_channels =  2 # T-V pair of values times the number of values per sequence
number_of_classes = boards

custom_learning_rate = 0.0001  # Leaning rate 0.001 by default
optimizer = Adam(learning_rate=custom_learning_rate)


input_shape = (sequence_length, number_of_channels)

# Network input
input_layer = Input(shape=input_shape)

# 1D Convolutional Layer
conv1d_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)

# MaxPooling Layer to reduce dimensionality
pooling_layer = MaxPooling1D(pool_size=2)(conv1d_layer)

# Flatten the output from the MaxPooling layer
flatten_layer = Flatten()(pooling_layer)

# Fully Connected (Dense) Layer
dense_layer = Dense(64, activation='relu')(flatten_layer)

# Output Layer
output_layer = Dense(number_of_classes + 1, activation='softmax')(dense_layer) # number_of_classes + 1 to include class otherwise excluded 
                                                                               # (Output Shape range is in the interval [0,number_of_classes + 1) 
                                                                               # and 0 is not used)
# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Model summary
model.summary()

#############################################################
#                                                           #
#                     TRAINING THE MODEL                    #
#                                                           #
#############################################################

# Early Stopping Implementation
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

num_epochs = 40  # Adjust the number of epochs 

# Training outside the loop
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, callbacks=[early_stopping])

# Save trained model in a HDF5 file
model.save("trained_model.h5")

# Save metrics
train_accuracy.extend(history.history['accuracy'])
val_accuracy.extend(history.history['val_accuracy'])
train_loss.extend(history.history['loss'])
val_loss.extend(history.history['val_loss'])

# Prepare x-axis
epochs = np.arange(1, len(train_accuracy) + 1 )


# Accuracy Plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Configurar ticks en el eje x con saltos de 2, asegurándonos de incluir la última época si es par
xticks_values = range(2, len(epochs) + 1, 2)

if len(epochs) % 2 == 0:
    xticks_values = list(xticks_values) + [len(epochs)]

plt.xticks(xticks_values, [str(int(epochs[i - 1])) for i in xticks_values])
plt.grid(True)
plt.xlim(1, len(epochs))
plt.ylim(0, 1)


# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.xticks(xticks_values, [str(int(epochs[i - 1])) for i in xticks_values])
plt.grid(True)
plt.xlim(1, len(epochs))

# Get model scores on the test set
y_scores = model.predict(x_test)
# Obtain predicted classes
y_pred = np.argmax(y_scores, axis=1)

# Calculate precision, recall, and F1-Score
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

# Print the metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


# Define classes
boards_indexes = list(range(1, 21))
# Create a DataFrame with metrics for each class
metrics_df = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1-Score': f1}, index=boards_indexes)
# Customize the style of the heatmap
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(metrics_df, annot=True, cmap='coolwarm', fmt='.3f', linewidths=.5)
plt.title('Metrics Heatmap for Each Board')
plt.xlabel('Metrics')
# Rotate y-axis labels by 90 degrees clockwise and add "Board" text
heatmap.set_yticklabels([f'Board {int(i+1)}' for i in heatmap.get_yticks()], rotation=0)

# Calculate the confusion matrix
cm  = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(16, 16))
classes = [f'Board {board}' for board in range(1, boards + 1)]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('\nPredicted Classes')
plt.ylabel('Real Classes')
plt.title('Confusion Matrix')
plt.show()

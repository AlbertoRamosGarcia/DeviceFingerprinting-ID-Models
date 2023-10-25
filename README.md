# Stimulated Microcontroller Dataset for New IoT Device Identification Schemes through On-Chip Sensor Monitoring

In this repository, the script used to train a 1D CNN-based model for device identification from sequences containing consecutive pairs of temperature and voltage sensor readings during various workloads is provided. They must be correctly labeled, following the workflow and using the tools described in [link to Zenodo/Tools].

The pre-trained model used in [link to the paper] is provided.

The ACQ1 subset utilized, in addition to the rest of the MOSID dataset, is accessible at: [link/DOI with the Dataset badge].

## Requirements

Before running the script, it will be necessary to import the following libraries:

 - Pandas
 - Tqdm
 - Matplotlib
 - Seaborn
 - NumPy
 - H5py
 - TensorFlow
 - Scikit-learn

## IDENTIFICATION MODEL

The trained model, "trained_model_1DCNN_20boards_LR0001.h5," has been trained for the multiclass classification of 20 boards, using a total of 1,364,800 sequences of T-V pairs with a length of 100 as the train/validation/test set (in a proportion of 70/15/15, respectively). The selected hyperparameters for training were a learning rate of 0.0001 and 40 epochs, with an 'early stop' callback of 4 epochs, while monitoring 'val_loss'.

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 100, 2)]          0

 conv1d (Conv1D)             (None, 98, 32)            224

 max_pooling1d (MaxPooling1  (None, 49, 32)            0
 D)

 flatten (Flatten)           (None, 1568)              0

 dense (Dense)               (None, 64)                100416

 dense_1 (Dense)             (None, 21)                1365

=================================================================
Total params: 102005 (398.46 KB)
Trainable params: 102005 (398.46 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

```

As outputs, in addition to the trained model, there are plots of accuracy and error loss for training versus validation. Predictions are also made with the test set, and the results are visualized in the form of a confusion matrix.

## Citing The Dataset

To be Completed.

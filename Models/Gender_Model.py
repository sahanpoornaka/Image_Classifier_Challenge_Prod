import datetime
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from sklearn.metrics import classification_report, confusion_matrix

# DataSet Paths
train_set_path = '/data/shared/sahan_dabarera/Image_Classification_Challange/Data/PKL_Data/Train_Data.pkl'
test_set_path = '/data/shared/sahan_dabarera/Image_Classification_Challange/Data/PKL_Data/Test_Data.pkl'

# Other Consntants
model_name = 'Gender_Model'
image_size = (224, 224, 3)
batch_size = 16
num_epochs = 1000
learning_rate = 0.0001

# Model Path
model_save_path = '/data/shared/sahan_dabarera/Image_Classification_Challange/Saved_Models/'+model_name+'/'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# load and iterate training dataset
train_data = pickle.load(open(train_set_path, 'rb'))
test_data = pickle.load(open(test_set_path, 'rb'))

X_train = train_data['X_train']
Y_train = train_data['Yg']

X_test = test_data['X_train']
Y_test = test_data['Yg']

# Print the Input Array Shapes
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("Y_train Shape:", Y_train.shape)
print("Y_test Shape:", Y_test.shape)

# Create The Model
base_model = tf.keras.applications.Xception(input_shape = image_size, include_top = False, weights = "imagenet")
base_model.trainable = False

model = Sequential([
        base_model,
        Flatten(),
         # tf.keras.layers.Dense(512, activation="relu"),
        # Dense(256, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
        ]
 )

opt = Adam(learning_rate=learning_rate)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])



# TensorBoard Log Directory
log_dir = "/data/shared/sahan_dabarera/Image_Classification_Challange/Logs/fit/"+ model_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train The Model
history = model.fit(X_train, Y_train, epochs = num_epochs, validation_steps=5, steps_per_epoch=16, validation_data=(X_test, Y_test), batch_size=batch_size, callbacks=[tensorboard_callback])

# Save The Model
model.save(model_save_path)

# Predictions
Y_pred = model.predict(X_test)
Y_pred = (Y_pred >=0.5).astype('int')

# Confusion Matrix
print("Confusion Matrix")
print("Horz (True) | Vert (Pred) | Woman, Man")
print(confusion_matrix(Y_test.astype('int'), Y_pred))

# Classification Report
print("Classification Report")
print(classification_report(Y_test.astype('int'), Y_pred, target_names = ['Woman (Class 0)', 'Man (Class 1)']))

# Print Parameters
print("Num Epochs:",num_epochs)
print("Learning Rate:", learning_rate)
print("Model Summary:", model.summary())
# You can run each step in jupyter notebook to have a better understand
# Jupyter notebook is highly recommended!

# Step 1 
# Load Dependencies

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix,accuracy_score

# Step 2 
# Setup variables
DATA_PATH = os.path.join('dataset_new')
#training dataset

# Actions that we try to detect
# Your dataset file should have the actions with the same order as the array
actions = np.array(['downward-dog','jumping-jack', 'leg-up', 'squat'])

# 30 videos worth of data
no_sequences = 30

# Sequence length
sl = 30

# label_map will look like : 
# {'downward-dog': 0, 'jumping-jack': 1, 'leg-up': 2, 'squat': 3}
label_map = {label:num for num, label in enumerate(actions)}

# Step 3 
# Load dataset from local file to numpy array
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sl):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)),allow_pickle=True)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# If you are using jupyter notebook, you can run
# np.array(sequences).shape and np.array(labels).shape to check the shape. 
# In my case: np.array(sequences).shape = (120,30,51) 
# = (4 actions x 30 sequences for each, 30 frams for each sequence, 51 keypoints for each frame)

# Step 4 
# Split the data to training and testing dataset with the ratio of 10%
X = np.array(sequences)
# This encode the action labels to one-hot code
# For example, [1,0,0,0] means 'downward-dog' 
# because 'downward-dog' is the 0th element of the array
y = to_categorical(labels).astype(int)
# If you use jupyter notebook, you can see that
# y equal to something like array([[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]....])

# The ratio can be changed in "test_size" variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Step 5 
# Build model

# With this 2 line you can get a visualized graph 
# of how accuracy is changing by using Tensorboard(VS code can directly get it)
# If you want to know more, please message me or google it by yourself
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Innitialize the Network
model1 = Sequential()
# Build up layers
model1.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,51)))
model1.add(LSTM(128, return_sequences=True, activation='relu'))
model1.add(LSTM(64, return_sequences=False, activation='relu'))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(actions.shape[0], activation='softmax'))

# Here, we use 'categorical_crossentropy' for the loss fuction 
# because we have multiple categories
# If there are only 2 categories, you should use 'binary_crossentropy'
model1.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Input training dataset
# You can play with the epochs to avoid overtraining.
model1.fit(X_train, y_train, epochs=250, callbacks=[tb_callback])

# This is the summary for training
model1.summary()

# Step 6
# Make Evaluation of our model
# Here is where we use testing dataset
yhat = model1.predict(X_test)

# Make confusion matrix to see how well the model is trained
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

# You may want to print out those 2 value to see how it look like
multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)

# Step 7
# Don't forget it save your model
# You should save your model weight if it is trained well
# The file should end with '.h5' extension
model1.save('something.h5')

# I will explain how to load your model in the real-time testing py file



'''
dataset:
----jumping-jack:
        ----0:
            ----0.npy(numpy array with length 51)
            ----1.npy
            ----2.npy
            ......
            ----29.npy
        ----1
        ----2
        ......
        --29 
----downward-dog
----squat       

'''





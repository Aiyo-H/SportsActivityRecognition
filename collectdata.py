# If you are using jupyter notebook, you can run each step one by one to make sure there is no error
# Try to understand the code from step 3 to the end

# Better to create a new folder and put this file in the folder
# Step 1

# Import all dependencies
# If one is missing just do: pip install packagename
# For the package cv2, to install it type: pip install opencv-python
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import csv
import pandas as pd
import os
import sys
import tempfile
import tqdm
from matplotlib.collections import LineCollection
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import imageio
from IPython.display import HTML, display

# Step 2

# I copy and pasted the code from the Movenet website of google
# You don't need to understand this part, or if you are curious, you can go to:
# https://www.tensorflow.org/hub/tutorials/movenet for more details
# Load helper functions for drawing and Movenet model for extracting keypoints
# Helper functions
# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

def to_gif(images, fps):
  """Converts image sequence (4D numpy array) to gif."""
  imageio.mimsave('./animation.gif', images, fps=fps)
  return embed.embed_file('./animation.gif')

def progress(value, max=100):
  return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))


# Load the Movenet lightning model
# You can use other versions of the model to see how it works
# Here we set the default to "movenet_lightning"
model_name = "movenet_lightning"

if "tflite" in model_name:
  if "movenet_lightning_f16" in model_name:
    os.system('!wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite')
    input_size = 192
  elif "movenet_thunder_f16" in model_name:
    os.system('!wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite')
    input_size = 256
  elif "movenet_lightning_int8" in model_name:
    os.system('!wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite')
    input_size = 192
  elif "movenet_thunder_int8" in model_name:
    os.system('!wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite')
    input_size = 256
  else:
    raise ValueError("Unsupported model name: %s" % model_name)

  # Initialize the TFLite interpreter
  interpreter = tf.lite.Interpreter(model_path="model.tflite")
  interpreter.allocate_tensors()

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

else:
  if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192
  elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
  else:
    raise ValueError("Unsupported model name: %s" % model_name)

  def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores


# Step 3 Important!

# Create folders for storing the extracted data
# Create a new folder in the current working directory called "dataset"
# You can change the name as you want
DATA_PATH = os.path.join('ddd') 

# Actions that we want to detect
# You can add more if you want
# But before you decide a new action to collect, please @ me in the slack. I will check if it is appropriate to put in the dataset
actions = np.array(['squat'])

# For each action we will have to record 30 different videos/sequences
no_sequences = 30

# For each sequence/video, there should be 30 frames for each
sequence_length = 30

# Create the folder
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
# You can check if the folder is created in your current working directory


# Step 4 Important!


# Collect data from the action you choose

# Specify the name of the action. 
# This action must be included in the folder previously created!
# For example, if you are collecting data for push-up, change the variable 'squat' to 'push-up'
# Here, I use squat for an example
# Only the folder with name you specifiy here will be filled with data, other folders will be empty
# You just need to give me the action folder, for example 'push-up' or any other actions
action = "squat"

# Get access to camera
cap = cv2.VideoCapture(0)

# If you changed the action, don't forget to also change here
if action == "squat":


    # For each sequence in 30 sequences
    # You can play with the loop here to collect data separately
    # For example, you can change it to:
        # for sequence in range(0,10):
        # In this way, you can collect data for the first 10 sequences instead of doing 30 push-ups in a roll.
    for sequence in range(no_sequences):

        # For each frame in 30 frames
        for frame_num in range(sequence_length):

            # Read feed
            ret, frame = cap.read()

            # display to screen
            # Calculate and display
            input_image = tf.expand_dims(frame, axis=0)
            input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
            # Run model inference and get raw keypoints tensor
            keypoints_with_scores = movenet(input_image)

            
            # Draw predictions on the image.
            display_image = tf.expand_dims(frame, axis=0)
            display_image = tf.cast(tf.image.resize_with_pad(
                display_image, 1280, 1280), dtype=tf.int32)
            # This output_overlay is the image with keypoints and connections drawed on it
            output_overlay = draw_prediction_on_image(
                np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)


            # Important! Please Read

            # After the program is been ran, a window will pop-up,
            # Or if it is not pop-up, it will appear in your task bar, 
            # open the windows ASAP so that you won't miss the data collection
            # As window shows "Pause for x secs", the collection is paused.
            # It will pause at the begining so that you can be prepared.
            # After 5 secs(this can be changed with waitkey function), the collection will start
            # SO, once the sentence "Pause for x secs" disappeared, Start to do the action!
            # Each pause means 1 sequence of 30 frames is been collected.
            # After 30 sequences, the program will stop automatically and the window will be close as well

            # Tips for doing action:
                # 1. Please stay in the center of the camera
                # 2. Please find a big space to do the action so that the camera can see all of your body and detect all keypoints
                # 3. The time for collect 1 sequence is depend on your computer performance,
                # you can try several times to see how fast you should do the action
                # But at least do one action for each sequence
                # 4. During the pause time, you can change the direction where you are facing to make the dataset more robust.
                # I will send more detail in the slack
                # 5. Please do not do any other movement during collection, or the dataset will mess up and ruin the network training
                # 6. If you accidently do any other movement that is irrelevent to the action, 
                # please delete the 'dataset' folder, run the program again, and do the action from the beginning
                # And anytime you interrupted the collection, please delete the 'dataset' folder, run the program again, and do the action from the beginning


            # Here is the sentences that will print on the screen
            # to inform you when to do the Action
            if frame_num == 0: 
                cv2.putText(output_overlay, 'Pause for 5 secs', (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(output_overlay, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', output_overlay)
                # Change the waiting time in miliseconds. e.g. 5000 msec = 5 sec
                cv2.waitKey(5000)
            else: 
                cv2.putText(output_overlay, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', output_overlay)
            # Save np.arrays in the folder
            keypoints = np.array(keypoints_with_scores.flatten())
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)
            
            # Break
        
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()



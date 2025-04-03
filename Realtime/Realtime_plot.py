import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pyomyo import Myo, emg_mode
from tensorflow.keras.models import load_model  # type: ignore
from collections import deque

# Sampling rate
SAMPLING_RATE = 50  # Hz

#Loading the trained model 
model_path = r"best_models\XLarge_model.h5"
try:
    model = load_model(model_path)
    print(f"[INFO] Model loaded from: {model_path}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit(1)

# Gesture classes in the same order as during training.
gesture_classes = ["Closed Fist", "Wrist Extension", "Wrist Flexion"]

# Updated window_size to 50 samples to match the training data.
window_size = 50  
emg_window = deque(maxlen=window_size)

# Smoothing buffer for predictions (last 10 predictions)
smoothing_buffer = deque(maxlen=10)

def preprocess_window(window):
   
    # Convert the deque to a NumPy array of shape (window_size, num_channels) where num_channels is 8.
    window_array = np.array(window, dtype=np.float32)
    
    # Append a column of zeros to change shape from (window_size, 8) to (window_size, 9)
    zeros_col = np.zeros((window_array.shape[0], 1), dtype=window_array.dtype)
    window_array = np.concatenate([window_array, zeros_col], axis=1)
    
    # Reshape to (1, window_size, num_channels), now with num_channels=9.
    input_data = window_array.reshape((1, window_size, -1))
    return input_data

plt.ion()  # Enable interactive mode
fig, ax = plt.subplots(figsize=(8, 6))

max_points = 100  # Maximum number of time steps to display

x_values = []             # Time steps
recognized_gestures = []  # Each element is an index corresponding to gesture_classes

(line,) = ax.plot([], [], marker='o', linestyle='-')

# Configure the y-axis to show gesture names.
ax.set_ylim(-0.5, len(gesture_classes) - 0.5)
ax.set_yticks(range(len(gesture_classes)))
ax.set_yticklabels(gesture_classes)
ax.set_xlim(0, max_points)
ax.set_xlabel('Time Steps')
ax.set_ylabel('Gesture')
title_text = ax.set_title('Current Gesture: None')

def update_plot(aggregated_prediction):
    confidence_str = ", ".join(
        [f"{gesture_classes[i]}: {aggregated_prediction[i]*100:.2f}%" for i in range(len(gesture_classes))]
    )
    title_text.set_text(f"Confidence: {confidence_str}")
    
    # Determine the recognized gesture based on the highest aggregated confidence.
    gesture_idx = np.argmax(aggregated_prediction)
    current_time = len(x_values)
    x_values.append(current_time)
    recognized_gestures.append(gesture_idx)
    
    # Keep only the latest max_points data points.
    if len(x_values) > max_points:
        x_display = x_values[-max_points:]
        y_display = recognized_gestures[-max_points:]
    else:
        x_display = x_values
        y_display = recognized_gestures
    
    line.set_data(x_display, y_display)
    if x_display:
        ax.set_xlim(x_display[0], x_display[-1])
    
    fig.canvas.draw()
    fig.canvas.flush_events()

def emg_handler(emg_values, movement):
    
    # Add the incoming sample (a tuple of sensor values) to the sliding window.
    emg_window.append(emg_values)
    
    # Process only if we have enough samples in the window.
    if len(emg_window) == window_size:
        # Preprocess the window to get input data in shape (1, 50, 9)
        input_data = preprocess_window(emg_window)
        predictions = model.predict(input_data)[0]  # shape: (num_classes,)
        
        # Add the prediction to the smoothing buffer.
        smoothing_buffer.append(predictions)
        
        # If the smoothing buffer is full, aggregate predictions and update the plot.
        if len(smoothing_buffer) == smoothing_buffer.maxlen:
            aggregated_prediction = np.mean(smoothing_buffer, axis=0)
            update_plot(aggregated_prediction)


#Initalise and connect Myo
myo_device = Myo(mode=emg_mode.PREPROCESSED)
try:
    myo_device.connect()
    print("[INFO] Myo device connected successfully.")
except Exception as e:
    print(f"[ERROR] Failed to connect to Myo device: {e}")
    exit(1)

# Attach the EMG handler.
myo_device.add_emg_handler(emg_handler)  # type: ignore

# Optionally set LED color and vibrate to indicate connection.
myo_device.set_leds([128, 128, 0], [128, 128, 0])
myo_device.vibrate(1)

print("[INFO] Streaming live EMG signals with real-time gesture categorical plot. Press Ctrl+C to stop.\n")

try:
    while True:
        myo_device.run()
        time.sleep(0.01)  # Sleep briefly to lower CPU usage.
except KeyboardInterrupt:
    print("\n[INFO] Stopping live EMG stream...")
finally:
    myo_device.disconnect()
    print("[INFO] Myo device disconnected.")

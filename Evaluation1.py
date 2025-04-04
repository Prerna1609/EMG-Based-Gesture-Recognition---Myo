import time
import numpy as np
import matplotlib
# Use a GUI backend (e.g., TkAgg) to enable interactive plotting in VSCode or externally
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pyomyo import Myo, emg_mode
from tensorflow.keras.models import load_model  # type: ignore
from collections import deque

# Adjust sampling rate if needed (the armband works around 50Hz)
SAMPLING_RATE = 50  # Hz

# 1. Load the Trained Model


model_path = r"C:\Users\Prerna\Desktop\Myo Test\best_models\XLarge_model.h5"
try:
    model = load_model(model_path)
    print(f"[INFO] Model loaded from: {model_path}")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit(1)

# Gesture classes in the same order as during training.
gesture_classes = ["Closed Fist", "Wrist Extension", "Wrist Flexion"]


# Global Buffers for Sliding Window, Smoothing, and History

# Updated window_size to 50 samples to match the training data.
window_size = 50  
emg_window = deque(maxlen=window_size)

# Smoothing buffer for predictions (e.g., last 10 predictions)
smoothing_buffer = deque(maxlen=10)

# Confidence history (timestamp, aggregated prediction) -- already in use
confidence_history = []  

# New: Recognized gesture history (timestamp, recognized gesture index)
recognized_history = []  


# 2. Preprocessing Function for the Sliding Window

def preprocess_window(window):
    """
    Given a sliding window of raw EMG samples (each sample is a tuple of sensor values),
    this function returns the raw window data as a numpy array of shape (1, window_size, num_channels),
    which is the expected input shape for our trained CNN model.
    
    The Myo device provides 8-channel data, but the model was trained with 9 channels.
    We append an extra column of zeros to match the expected input shape.
    """
    # Convert the deque to a NumPy array of shape (window_size, num_channels) where num_channels is 8.
    window_array = np.array(window, dtype=np.float32)
    
    # Append a column of zeros to change shape from (window_size, 8) to (window_size, 9)
    zeros_col = np.zeros((window_array.shape[0], 1), dtype=window_array.dtype)
    window_array = np.concatenate([window_array, zeros_col], axis=1)
    
    # Reshape to (1, window_size, num_channels), now with num_channels=9.
    input_data = window_array.reshape((1, window_size, -1))
    return input_data


# 3. Real-Time Categorical Plot Setup

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
title_text = ax.set_title('Current Confidence: None')

# New: Instruction text to show which gesture the user should display.
instruction_text = ax.text(0.5, 0.95, "", transform=ax.transAxes, ha="center", fontsize=14, color="red")


# 4. Update Plot Function (also record history)

def update_plot(aggregated_prediction):
    """
    Update the line plot with the recognized gesture and display the confidence scores.
    Record both the confidence and the recognized gesture (highest confidence).
    """
    confidence_str = ", ".join(
        [f"{gesture_classes[i]}: {aggregated_prediction[i]*100:.2f}%" for i in range(len(gesture_classes))]
    )
    title_text.set_text(f"Confidence: {confidence_str}")
    
    # Record the current aggregated prediction with timestamp.
    confidence_history.append((time.time(), aggregated_prediction.copy()))
    
    # Determine the recognized gesture based on the highest aggregated confidence.
    gesture_idx = np.argmax(aggregated_prediction)
    recognized_history.append((time.time(), gesture_idx))
    
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

# 5. EMG Handler Function with Sliding Window and Smoothing

def emg_handler(emg_values, movement):
    """
    Called every time the Myo device receives a new EMG sample.
    Uses a sliding window to collect raw EMG samples and predict the gesture.
    A smoothing buffer averages the predictions before updating the plot.
    """
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

# 6. Initialize & Connect Myo

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

# 7. Setup for 1-Minute Run with 10-Second Intervals and Expected Gestures

run_duration = 120  # seconds
interval_duration = 20  # seconds
# Define expected sequence (using indices into gesture_classes)
expected_sequence = [0, 1, 2, 0, 1, 2]  # For 6 intervals of 10 seconds each.
current_interval = 0
current_expected_gesture = expected_sequence[current_interval]
instruction_text.set_text(f"Show: {gesture_classes[current_expected_gesture]}")
print(f"[INFO] Interval 1: Please show {gesture_classes[current_expected_gesture]}")

start_time = time.time()


# 8. Main Loop for Real-Time Streaming for One Minute

try:
    while time.time() - start_time < run_duration:
        # Update expected gesture every interval_duration seconds.
        new_interval = int((time.time() - start_time) // interval_duration)
        if new_interval != current_interval and new_interval < len(expected_sequence):
            current_interval = new_interval
            current_expected_gesture = expected_sequence[current_interval]
            instruction_text.set_text(f"Show: {gesture_classes[current_expected_gesture]}")
            print(f"[INFO] Interval {current_interval+1}: Please show {gesture_classes[current_expected_gesture]}")
        myo_device.run()
        time.sleep(0.01)  # Sleep briefly to lower CPU usage.
except KeyboardInterrupt:
    print("\n[INFO] Stopping live EMG stream...")
finally:
    myo_device.disconnect()
    print("[INFO] Myo device disconnected.")


# 9. Analyze Results for Each 10-Second Interval

accuracy_results = []
for i in range(len(expected_sequence)):
    interval_start = start_time + i * interval_duration
    interval_end = start_time + (i+1) * interval_duration
    # Filter recognized_history for predictions in this interval.
    predictions = [gesture for (t, gesture) in recognized_history if interval_start <= t < interval_end]
    if predictions:
        correct = sum(1 for p in predictions if p == expected_sequence[i])
        acc = correct / len(predictions)
    else:
        acc = None
    accuracy_results.append(acc)

print("\n[INFO] Interval-wise Accuracy:")
for i, acc in enumerate(accuracy_results):
    if acc is not None:
        print(f"Interval {i+1} ({gesture_classes[expected_sequence[i]]}): {acc*100:.2f}%")
    else:
        print(f"Interval {i+1} ({gesture_classes[expected_sequence[i]]}): No data recorded.")

# Optionally, compute overall accuracy across intervals where data is available.
valid_accuracies = [acc for acc in accuracy_results if acc is not None]
if valid_accuracies:
    overall_acc = np.mean(valid_accuracies)
    print(f"\n[INFO] Overall Accuracy: {overall_acc*100:.2f}%")
else:
    print("\n[INFO] No accuracy data available.")

# Optionally, plot recognized gesture history with expected gesture intervals.
plt.ioff()
fig2, ax2 = plt.subplots(figsize=(10, 6))
# Plot recognized gestures over time
times = [t - start_time for (t, _) in recognized_history]
recognized_vals = [gesture for (_, gesture) in recognized_history]
ax2.plot(times, recognized_vals, marker='o', linestyle='-', label="Recognized Gesture")
# Plot expected gesture as horizontal lines for each interval.
for i, expected in enumerate(expected_sequence):
    ax2.hlines(expected, i*interval_duration, (i+1)*interval_duration, colors='red', linestyles='dashed', label="Expected Gesture" if i==0 else "")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Gesture (Index)")
ax2.set_title("Recognized vs Expected Gestures Over 1 Minute")
ax2.legend()
plt.show()

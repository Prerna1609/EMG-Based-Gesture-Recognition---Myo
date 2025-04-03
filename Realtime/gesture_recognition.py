import time
import numpy as np
from collections import deque
from pyomyo import Myo, emg_mode
from tensorflow.keras.models import load_model

class GestureRecognizer:
    def __init__(self, model_path="best_models\XLarge_model.h5", window_size=50, smoothing_buffer_size=10):
        self.window_size = window_size
        self.smoothing_buffer_size = smoothing_buffer_size
        
        # Load the trained model.
        try:
            self.model = load_model(model_path)
            print(f"[INFO] Model loaded from: {model_path}")
        except Exception as e:
            print(f"[ERROR] Could not load model: {e}")
            exit(1)
        
        # Define gesture classes in the same order as used during training.
        self.gesture_classes = ["Closed Fist", "Wrist Extension", "Wrist Flexion"]
        
        # Create sliding windows for EMG samples and predictions.
        self.emg_window = deque(maxlen=self.window_size)
        self.smoothing_buffer = deque(maxlen=self.smoothing_buffer_size)
        
        # Initialize and connect the Myo device.
        self.myo_device = Myo(mode=emg_mode.PREPROCESSED)
        try:
            self.myo_device.connect()
            print("[INFO] Myo device connected successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to connect to Myo device: {e}")
            exit(1)
        
        # Attach the EMG handler.
        self.myo_device.add_emg_handler(self.emg_handler)
        
        # Optionally set LED color and vibrate to indicate connection.
        self.myo_device.set_leds([128, 128, 0], [128, 128, 0])
        self.myo_device.vibrate(1)
        
        print("[INFO] Gesture recognition started. Press Ctrl+C to stop.\n")
    
    def preprocess_window(self, window):
        """
        Convert the deque window into a NumPy array suitable for prediction.
        The original data has 8 channels; a column of zeros is appended to match
        the expected input shape of (window_size, 9).
        """
        window_array = np.array(window, dtype=np.float32)
        zeros_col = np.zeros((window_array.shape[0], 1), dtype=window_array.dtype)
        window_array = np.concatenate([window_array, zeros_col], axis=1)
        input_data = window_array.reshape((1, self.window_size, -1))
        return input_data

    def emg_handler(self, emg_values, movement):
        """
        Handles incoming EMG data, aggregates samples into a window, and once
        enough data is collected, makes a prediction.
        """
        self.emg_window.append(emg_values)
        if len(self.emg_window) == self.window_size:
            input_data = self.preprocess_window(self.emg_window)
            # Suppress verbose output by setting verbose=0.
            predictions = self.model.predict(input_data, verbose=0)[0]
            self.smoothing_buffer.append(predictions)
            
            # Only process when smoothing buffer is full.
            if len(self.smoothing_buffer) == self.smoothing_buffer.maxlen:
                aggregated_prediction = np.mean(self.smoothing_buffer, axis=0)
                self.handle_prediction(aggregated_prediction)

    def handle_prediction(self, aggregated_prediction):
        """
        Determines the gesture with the highest confidence and prints it.
        """
        gesture_idx = np.argmax(aggregated_prediction)
        current_gesture = self.gesture_classes[gesture_idx]
        print("Current gesture:", current_gesture)

    def run(self):
        """
        Main loop to keep processing EMG data.
        """
        try:
            while True:
                self.myo_device.run()
                time.sleep(0.01)  # Brief sleep to lower CPU usage.
        except KeyboardInterrupt:
            print("\n[INFO] Stopping gesture recognition...")
        finally:
            self.myo_device.disconnect()
            print("[INFO] Myo device disconnected.")


if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.run()

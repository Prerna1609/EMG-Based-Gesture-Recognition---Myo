import tkinter as tk
from tkinter import font
import time
import threading
import queue
import numpy as np
from pyomyo import Myo, emg_mode
from tensorflow.keras.models import load_model  # type: ignore
from collections import deque
import pandas as pd
from PIL import Image, ImageTk  # For loading and displaying images


# Global Settings & Model Initialization

SAMPLING_RATE = 50  # Hz
window_size = 50   # 50 samples ~ 1 second at 50Hz
emg_window = deque(maxlen=window_size)
# Reduced smoothing buffer size from 5 to 3 for quicker responsiveness
smoothing_buffer = deque(maxlen=3)

# Set confidence threshold to 0.75
confidence_threshold = 0.75

# Gesture classes in the same order as during training.
gesture_classes = ["Closed Fist", "Wrist Extension", "Wrist Flexion"]

# Load the trained model.
model_path = r"best_models\XLarge_model.h5"
try:
    model = load_model(model_path)
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit(1)


# Therapy Session Setup & Global Variables

total_session_time = 0      # in seconds
gesture_hold_time = 0       # in seconds
session_start_time = 0      # record the session start time

exercises = []  
current_exercise_idx = 0
in_correct_gesture = False
correct_gesture_start_time = 0.0
session_active = True
in_break = False           
error_mode = False          

# Variables to help manage multiple sessions
session_counter = 0
myo_device_global = None  # Global reference to Myo device

# Image Paths for Each Gesture
gesture_images = {
    "Closed Fist":      r"C:\Users\Prerna\Desktop\Myo Test\Images\Closed Fist.jpeg",
    "Wrist Extension":  r"C:\Users\Prerna\Desktop\Myo Test\Images\Wrist Extension.jpeg",
    "Wrist Flexion":    r"C:\Users\Prerna\Desktop\Myo Test\Images\Wrist Flexion.jpeg"
}

# GUI Queue for Thread-Safe Updates
gui_queue = queue.Queue()

def send_gui_message(msg, tag="default"):
    gui_queue.put((tag, msg))

# Data Preprocessing and Feature Extraction

def preprocess_window(window):
    window_array = np.array(window, dtype=np.float32)
    zeros_col = np.zeros((window_array.shape[0], 1), dtype=window_array.dtype)
    window_array = np.concatenate([window_array, zeros_col], axis=1)
    input_data = window_array.reshape((1, window_size, -1))
    return input_data

def wave_form_length(raw_data):
    rows = []
    size = 50
    slices = int(raw_data.shape[0] / size)
    for sample in range(slices):
        window = raw_data.loc[(sample * size):(sample * size) + (size - 1)]
        wl_values = (abs(window.diff()).sum()) / size
        rows.append(wl_values)
    temp_list = pd.DataFrame(rows)
    temp_list.columns = ['wl1','wl2','wl3','wl4','wl5','wl6','wl7','wl8']
    return temp_list

# Utility: Update the Gesture Image on the Right Panel

def show_exercise_image(gesture):
    path = gesture_images.get(gesture)
    print("Loading image from path:", path)  # Debug output
    if path is not None:
        try:
            img = Image.open(path)
            # Apply rotation based on gesture type:
            if gesture == "Closed Fist":
                img = img.rotate(-90, expand=True)
            elif gesture in ["Wrist Extension", "Wrist Flexion"]:
                img = img.rotate(180, expand=True)
            # Resize the image to 600x600 pixels
            img = img.resize((600, 600), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            gesture_image_label.config(image=tk_img)
            gesture_image_label.image = tk_img  # Keep a reference to avoid garbage collection
        except Exception as e:
            gesture_image_label.config(text=f"Could not load image for {gesture}\n{e}")
    else:
        gesture_image_label.config(text=f"No image found for '{gesture}'")

# Break Resume Function

def resume_next_exercise():
    global in_break, current_exercise_idx, session_active
    in_break = False
    if current_exercise_idx < len(exercises):
        next_gesture, next_duration = exercises[current_exercise_idx]
        send_gui_message(f"Break over. Next, perform '{next_gesture}' for {next_duration} sec.", tag="instruction")
        show_exercise_image(next_gesture)
    else:
        send_gui_message("Therapy session complete, thank you!", tag="instruction")
        session_active = False

# Session Update Logic (called with each aggregated prediction)

def update_session(aggregated_prediction):
    global current_exercise_idx, in_correct_gesture, correct_gesture_start_time, session_active, in_break, error_mode
    if in_break or error_mode:
        return

    if np.max(aggregated_prediction) < confidence_threshold:
        return

    gesture_idx = np.argmax(aggregated_prediction)
    recognized_gesture = gesture_classes[gesture_idx]
    expected_gesture, hold_duration = exercises[current_exercise_idx]
    
    send_gui_message(f"Expected: '{expected_gesture}' | Recognized: '{recognized_gesture}'", tag="instruction")
    
    if recognized_gesture == expected_gesture:
        if not in_correct_gesture:
            in_correct_gesture = True
            correct_gesture_start_time = time.time()
            send_gui_message(f"Start performing '{expected_gesture}' now.", tag="instruction")
        else:
            elapsed = time.time() - correct_gesture_start_time
            send_gui_message(f"Holding '{expected_gesture}': {elapsed:.1f}/{hold_duration} sec", tag="instruction")
            if elapsed >= hold_duration:
                send_gui_message(f"Well done! You held '{expected_gesture}' for {hold_duration} sec.", tag="instruction")
                current_exercise_idx += 1
                in_correct_gesture = False
                smoothing_buffer.clear()
                
                if current_exercise_idx < len(exercises):
                    send_gui_message("Take a 5-second break...", tag="instruction")
                    in_break = True
                    threading.Timer(5, resume_next_exercise).start()
                else:
                    send_gui_message("Therapy session complete, thank you!", tag="instruction")
                    session_active = False
    else:
        if in_correct_gesture:
            error_mode = True
            send_gui_message("Exercise interrupted. Please restart the exercise.", tag="instruction")
            
            def clear_error():
                global error_mode, correct_gesture_start_time
                error_mode = False
                correct_gesture_start_time = time.time()
            
            threading.Timer(2, clear_error).start()  # Reduced error timer to 2 seconds
        else:
            send_gui_message(f"Please perform '{expected_gesture}'.", tag="instruction")
        in_correct_gesture = False

# EMG Handler & Myo Streaming

def emg_handler(emg_values, movement):
    emg_window.append(emg_values)
    if len(emg_window) == window_size:
        input_data = preprocess_window(emg_window)
        predictions = model.predict(input_data)[0]
        smoothing_buffer.append(predictions)
        if len(smoothing_buffer) == smoothing_buffer.maxlen:
            aggregated_prediction = np.mean(smoothing_buffer, axis=0)
            update_session(aggregated_prediction)
        # Uncomment if you wish to display waveform length information:
        # raw_df = pd.DataFrame(np.array(emg_window), columns=['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8'])
        # wl_features = wave_form_length(raw_df)
        # send_gui_message(f"Waveform Length: {wl_features.iloc[0].values}", tag="default")

def run_myo():
    global session_active, myo_device_global
    myo_device = Myo(mode=emg_mode.PREPROCESSED)
    myo_device_global = myo_device  # Save reference
    try:
        myo_device.connect()
        send_gui_message("[INFO] Myo device connected successfully.", tag="info")
    except Exception as e:
        send_gui_message(f"[ERROR] Failed to connect to Myo device: {e}", tag="error")
        exit(1)
    
    myo_device.add_emg_handler(emg_handler)
    myo_device.set_leds([128, 128, 0], [128, 128, 0])
    myo_device.vibrate(1)
    send_gui_message("[INFO] Therapy session in progress. Perform the gestures as instructed.", tag="info")
    
    try:
        while session_active:
            myo_device.run()
            time.sleep(0.01)
    finally:
        # Explicitly close the serial port connection when done.
        myo_device.disconnect()  # This should internally close the port.
        myo_device_global = None
        send_gui_message("[INFO] Myo device disconnected.", tag="info")

# Tkinter GUI Setup with Enhanced Styling and an Expanded Image Panel
root = tk.Tk()
root.title("Therapy Session")
root.geometry("1200x700")  # Increased window size for more space
root.configure(bg="#f0f0f0")

# Header Frame
header_frame = tk.Frame(root, bg="#006064")
header_frame.pack(fill=tk.X)
header_label = tk.Label(header_frame, text="Therapy Session", font=("Helvetica", 24, "bold"), bg="#006064", fg="white")
header_label.pack(pady=10)

# Configuration Frame (User Inputs)
config_frame = tk.Frame(root, bg="#f0f0f0")
config_frame.pack(pady=10)
tk.Label(config_frame, text="Total Session Length (minutes):", font=("Helvetica", 14), bg="#f0f0f0").grid(row=0, column=0, padx=5, pady=5, sticky="e")
total_session_entry = tk.Entry(config_frame, font=("Helvetica", 14))
total_session_entry.grid(row=0, column=1, padx=5, pady=5)
tk.Label(config_frame, text="Gesture Hold Time (seconds):", font=("Helvetica", 14), bg="#f0f0f0").grid(row=1, column=0, padx=5, pady=5, sticky="e")
gesture_time_entry = tk.Entry(config_frame, font=("Helvetica", 14))
gesture_time_entry.grid(row=1, column=1, padx=5, pady=5)

# Main Content Frame
content_frame = tk.Frame(root, bg="#f0f0f0")
content_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

# Instruction Panel (Left Side)
instruction_frame = tk.Frame(content_frame, bg="#ffffff", bd=2, relief="groove")
instruction_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
instruction_label = tk.Label(instruction_frame, text="Instructions will appear here", font=("Helvetica", 20), bg="#ffffff", fg="#004d40", wraplength=350, justify="center")
instruction_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

# Log Panel (Below Instruction Panel) - Smaller height
log_frame = tk.Frame(content_frame, bg="#ffffff", bd=2, relief="groove")
log_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
log_text = tk.Text(log_frame, wrap=tk.WORD, font=("Helvetica", 12), bg="#ffffff", fg="black", height=5)
log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar = tk.Scrollbar(log_frame, command=log_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
log_text.config(yscrollcommand=scrollbar.set)

# Gesture Image Panel (Right Side) - Bigger panel now
gesture_image_frame = tk.Frame(content_frame, bg="#ffffff", bd=2, relief="groove", width=700, height=700)
gesture_image_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)
gesture_image_label = tk.Label(gesture_image_frame, text="Gesture Image\nwill appear here", font=("Helvetica", 14), bg="#ffffff", fg="black")
gesture_image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=1)
content_frame.grid_rowconfigure(0, weight=3)
content_frame.grid_rowconfigure(1, weight=1)

# Start Button Frame
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=10)
start_button = tk.Button(button_frame, 
                         text="Start Therapy Session", 
                         command=lambda: threading.Thread(target=start_session, daemon=True).start(), 
                         font=("Helvetica", 16), 
                         bg="#00796B", 
                         fg="black", 
                         bd=3, 
                         relief="raised")
start_button.pack()

# GUI Message Processing

def process_gui_queue():
    while not gui_queue.empty():
        tag, msg = gui_queue.get()
        if tag == "instruction":
            instruction_label.config(text=msg)
        else:
            log_text.insert(tk.END, msg + "\n")
            log_text.see(tk.END)
    root.after(100, process_gui_queue)

# Session Startup Routine

def start_session():
    global total_session_time, gesture_hold_time, exercises, session_start_time, current_exercise_idx, session_active, session_counter, myo_device_global
    try:
        total_minutes = float(total_session_entry.get())
        gesture_hold_time = float(gesture_time_entry.get())
    except ValueError:
        send_gui_message("Invalid input. Please enter numeric values.", tag="error")
        return
    
    total_session_time = total_minutes * 60
    exercises = [
        ("Closed Fist", gesture_hold_time),
        ("Wrist Extension", gesture_hold_time),
        ("Wrist Flexion", gesture_hold_time)
    ]
    current_exercise_idx = 0
    session_active = True
    session_start_time = time.time()
    
    global session_counter
    if session_counter > 0:
        send_gui_message("Please wait while the device resets...", tag="info")
        time.sleep(5)
    session_counter += 1
    
    config_frame.pack_forget()
    
    instruction_label.config(text="Welcome to the therapy session.\nPlease relax and take a deep breath.\nPlace your hand on the table for support.")
    time.sleep(3)
    instruction_label.config(text="Let's begin the therapy session in:")
    for count in range(3, 0, -1):
        instruction_label.config(text=f"{count}")
        time.sleep(1)
    instruction_label.config(text="Get ready by wearing the armband...")
    time.sleep(2)
    
    first_gesture, first_duration = exercises[current_exercise_idx]
    instruction_label.config(text=f"Please perform '{first_gesture}' for {first_duration} sec.")
    show_exercise_image(first_gesture)
    
    threading.Thread(target=run_myo, daemon=True).start()

process_gui_queue()
root.mainloop()

# coral_receiver/receive_and_infer.py

import socket
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict

from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

import cvzone  # For pretty overlays

SOCKET_PATH = "../socket_interface/camera_socket.sock"

# Load the COCO class list from a text file
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

model = YOLO('240_yolov8n_full_integer_quant_edgetpu.tflite', task='detect')

# Load Coral model
interpreter = Interpreter(
    model_path="240_yolov8n_full_integer_quant_edgetpu.tflite",
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# Connect to camera sender
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect(SOCKET_PATH)
print("Connected to camera sender.")

# Initialize variables for tracking frame count and time
frame_count = 0
start_time = time.time()

# Variables to track total inference time and the number of frames processed for inference
total_inference_time = 0
inference_frame_count = 0

def receive_frame(sock):
    # Read size (4 bytes)
    size_data = sock.recv(4)
    if not size_data:
        return None
    size = int.from_bytes(size_data, byteorder='big')

    # Read image data
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet

    img_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return frame

try:
    while True:
        # Read the current frame from the video stream
        frame = receive_frame(client)
        if frame is None:
            print("Error: unable to capture frame")
            break

        frame_count += 1
        # if frame_count % 6 != 0:
            # continue  # Skip every 5 frames to improve performance (infer on every 6th frame)

        # Resize the frame to 240p (320x240) for faster processing
        resized_frame = cv2.resize(frame, (320, 240))

        # Start timer for the inference process
        inference_start = time.time()

        # Run YOLO prediction on the resized frame (240p)
        results = model.predict(resized_frame, imgsz=240)

        # End timer for the inference
        inference_end = time.time()
        total_inference_time += (inference_end - inference_start)
        inference_frame_count += 1

        # Dictionary to count detected objects by class
        label_count = defaultdict(int)

        # Check if there are any detection results
        if len(results) > 0:
            a = results[0].boxes.data
            if a is not None and len(a) > 0:
                px = pd.DataFrame(a).astype("float")

                # Loop through the detection results and draw rectangles around detected objects
                for index, row in px.iterrows():
                    # Convert the bounding box coordinates back to the original frame size
                    x1 = int(row[0] * (frame.shape[1] / 320))
                    y1 = int(row[1] * (frame.shape[0] / 240))
                    x2 = int(row[2] * (frame.shape[1] / 320))
                    y2 = int(row[3] * (frame.shape[0] / 240))
                    d = int(row[5])
                    c = class_list[d]

                    # Draw rectangles around detected objects and display the class label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

                    # Increment the count for the detected class
                    label_count[c] += 1

        # Calculate and display FPS (frames per second) for the visual stream
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        cvzone.putTextRect(frame, f'FPS (visual): {round(fps, 2)}', (10, 30), 1, 1)

        # Calculate and display FPS for the inference process
        if inference_frame_count > 0:
            inference_fps = inference_frame_count / total_inference_time
            cvzone.putTextRect(frame, f'FPS (inference): {round(inference_fps, 2)}', (10, 60), 1, 1)

        # Display the count of detected objects per class on the frame
        y_offset = 90  # Initial position for the text
        for label, count in label_count.items():
            cvzone.putTextRect(frame, f'{count} {label}', (10, y_offset), 1, 1)
            y_offset += 30  # Move the text down for each new label
        
        frame_color=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Show the frame with detections, FPS, and object counts
        cv2.imshow("FRAME", frame_color)

        # Exit the loop if the ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Stopping inference receiver.")
finally:
    client.close()
    cv2.destroyAllWindows()

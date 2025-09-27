# camera_sender/send_camera_frames.py
import socket
import os
import cv2
import numpy as np
from picamera2 import Picamera2
from io import BytesIO
from libcamera import Transform


SOCKET_PATH = "../socket_interface/camera_socket.sock"

# Remove socket file if it already exists
if os.path.exists(SOCKET_PATH):
    os.remove(SOCKET_PATH)

# Setup Unix socket server
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
server.listen(1)
print("Waiting for Coral inference client to connect...")

conn, _ = server.accept()
print("Client connected!")

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}, transform=Transform(hflip=True, vflip=True)))
picam2.start()

try:
    while True:
        frame = picam2.capture_array()
        
        # Convert to JPEG to reduce size
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        
        data = jpeg.tobytes()
        size = len(data)

        # Send size first (fixed 4-byte length header)
        conn.sendall(size.to_bytes(4, byteorder='big'))
        # Then send the image data
        conn.sendall(data)

except KeyboardInterrupt:
    print("Stopping camera sender.")
finally:
    conn.close()
    server.close()
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)

import cv2
import os

video_path = "data/video_data/sample.mp4"
output_folder = "data/video_data/frames"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (224, 224))

    # Save frame
    cv2.imwrite(f"{output_folder}/frame_{count}.jpg", frame)

    count += 1

cap.release()
print(f"Extracted {count} frames")
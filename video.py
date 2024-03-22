import cv2
import numpy as np
import mediapipe as mp
import time


MODEL_PATH1 = 'model/model_1.tflite'
MODEL_PATH2 = 'model/model.tflite'

options_1 = mp.tasks.vision.ObjectDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH1),
    max_results=6,
    running_mode=mp.tasks.vision.RunningMode.VIDEO)

options_2 = mp.tasks.vision.ObjectDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH2),
    max_results=6,
    running_mode=mp.tasks.vision.RunningMode.IMAGE)

print(mp.__version__)

detector_1 = mp.tasks.vision.ObjectDetector.create_from_options(options_1)
detector_2 = mp.tasks.vision.ObjectDetector.create_from_options(options_2)

# Load video with OpenCV.
video = cv2.VideoCapture('tiger_walking.mp4')
video_fps = video.get(cv2.CAP_PROP_FPS)
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) * 2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter('output_ultralytics.mp4', fourcc, 5.0, (frame_width, frame_height))

frame_index = 0
while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # Convert the BGR image to RGB and then to MediaPipe Image format.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    img_copy = np.copy(mp_frame.numpy_view())

    # Calculate the timestamp for the current frame in microseconds.
    frame_timestamp_us = int(1000000 * frame_index / video_fps)

    # Perform object detection using the timestamp in microseconds.
    detection_result = detector_1.detect_for_video(mp_frame, frame_timestamp_us)

    # Process detection results, draw bounding boxes and labels, etc.
    # Draw bounding boxes and labels on the frame.

    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

        # Prepare label text.
        label = detection.categories[0].category_name if detection.categories else "Unknown"
        score = detection.categories[0].score if detection.categories else 0
        label_text = f"{label}: {score:.2f}"

        #timestamp_incre = 1
        if score > 0.5:
            # Draw bounding box on the frame.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Put label text on the frame.
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cropped_frame = img_copy[y:y+h, x:x+w]
            cropped_frame = cropped_frame.astype(np.uint8)
            cropped_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_frame)
            body_result = detector_2.detect(cropped_frame)

            for body_part in body_result.detections:
                label = body_part.categories[0].category_name if body_part.categories else "Unknown"
                score_ = body_part.categories[0].score
                if label == 'leg':
                    label = 'torso'
                elif label == 'torso':
                    label == 'leg'
                label_text = f"{label}: {score_:.2f}"
                if score_ > 0.:
                    cropped_bbox = body_part.bounding_box
                    start_p = x + cropped_bbox.origin_x, y + cropped_bbox.origin_y
                    end_p = start_p[0] + cropped_bbox.width, start_p[1] + cropped_bbox.height
                    cv2.rectangle(frame, start_p, end_p, (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (start_p[0], start_p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #timestamp_incre += 1


    out.write(frame)
    frame_index += 1

    # Display the frame with bounding boxes.
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()

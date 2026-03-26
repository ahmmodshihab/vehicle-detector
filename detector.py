
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8s.pt')       

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}


def detect_vehicles(frame, line_position=0.5):
    """যেকোনো frame এ vehicle detect করবে"""
    height, width = frame.shape[:2]
    line_y = int(height * line_position)

    results = model(
        frame,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=0.25,                 # confidence threshold
        verbose=False
    )

    counts = {'car': 0, 'motorcycle': 0, 'bus': 0, 'truck': 0}
    crossed = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        vehicle_type = VEHICLE_CLASSES[class_id]
        counts[vehicle_type] += 1

        # Box এর center y position
        box_center_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

        # Line crossing check
        if abs(box_center_y - line_y) < 10:
            crossed.append(vehicle_type)

    annotated_frame = results[0].plot()

    # Line draw
    cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 255), 2)

    return annotated_frame, counts, crossed, results


def estimate_speed(box_prev, box_curr, fps, real_width_meters=4.0):
    """দুটো frame এর box position থেকে speed বের করবে"""
    if box_prev is None or box_curr is None:
        return 0.0

    prev_center_x = (box_prev[0] + box_prev[2]) / 2
    curr_center_x = (box_curr[0] + box_curr[2]) / 2

    pixel_movement = abs(curr_center_x - prev_center_x)

    box_width_pixels = box_curr[2] - box_curr[0]
    if box_width_pixels == 0:
        return 0.0

    meters_per_pixel = real_width_meters / box_width_pixels
    distance_meters = pixel_movement * meters_per_pixel

    speed = (distance_meters * fps) * 3.6
    return round(speed, 1)


def detect_from_image(image_bytes):
    """Image bytes থেকে detect করবে"""
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    annotated_frame, counts, crossed, _ = detect_vehicles(frame)
    return annotated_frame, counts


def detect_from_video(video_path, line_position=0.5, frame_skip=3):
    """Video file থেকে frame by frame detect করবে"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frames = []
    all_counts = []
    speeds = []
    prev_boxes = {}
    frame_idx = 0                  # frame counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # প্রতি frame_skip frame এ একবার detect করো
        if frame_idx % frame_skip != 0:
            continue

        annotated, counts, crossed, results = detect_vehicles(frame, line_position)

        # Speed estimation
        current_boxes = {}
        for i, box in enumerate(results[0].boxes):
            box_coords = box.xyxy[0].tolist()
            current_boxes[i] = box_coords
            speed = estimate_speed(
                prev_boxes.get(i),
                box_coords,
                fps
            )
            if speed > 0:
                speeds.append(speed)

        prev_boxes = current_boxes
        frames.append(annotated)
        all_counts.append(counts)

    cap.release()
    avg_speed = round(sum(speeds) / len(speeds), 1) if speeds else 0
    return frames, all_counts, avg_speed


def run_webcam(line_position=0.5):
    """Webcam থেকে real-time detect করবে"""
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    prev_boxes = {}
    print("Running... Press 'q' to stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, counts, crossed, results = detect_vehicles(frame, line_position)

        # Speed estimation
        for i, box in enumerate(results[0].boxes):
            box_coords = box.xyxy[0].tolist()
            speed = estimate_speed(prev_boxes.get(i), box_coords, fps)
            prev_boxes[i] = box_coords

            if speed > 0:
                x1, y1 = int(box_coords[0]), int(box_coords[1])
                cv2.putText(annotated, f'{speed} km/h', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # show counts on the frame
        y_pos = 40
        for vehicle, count in counts.items():
            cv2.putText(annotated, f'{vehicle}: {count}',
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 0), 2)
            y_pos += 35

        cv2.imshow('Vehicle Detection', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()
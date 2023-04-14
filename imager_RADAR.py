#Author: Tahrim Imon
#Date: April 13 (My Bday)
#Title: Recognition and Detection of Aircraft in Real-time (RADAR)

import cv2
import sys

def detect_aircraft(video_path):
    # Load YOLOv4 model
    model_weights = "yolov4.weights"
    model_config = "yolov4.cfg"
    model = cv2.dnn.readNet(model_weights, model_config)

    # Load COCO dataset classes
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        height, width, _ = frame.shape

        # Detect objects in the frame
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        model.setInput(blob)
        output_layers = model.getUnconnectedOutLayersNames()
        layer_outputs = model.forward(output_layers)

        # Process the detected objects
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] == "airplane":
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # Draw a green bounding box around the detected aircraft
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Aircraft Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python aircraft_detection.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    detect_aircraft(video_path)

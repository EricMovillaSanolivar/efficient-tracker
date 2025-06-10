import cv2
import time
import json
import base64
import requests
import threading
import mediapipe as mp
from mtracker import Mtracker
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load yolo classes equivalent
yolo_cls = None
with open("./classes.json", "r") as file:
    yolo_cls = json.load(file)

# Define model path
vmodel_path = python.BaseOptions(model_asset_path='./models/efficientdet_lite2.tflite', delegate=python.BaseOptions.Delegate.CPU)
# Define model options
voptions = vision.ObjectDetectorOptions(base_options=vmodel_path, score_threshold=0.3, max_results=100)
# Reference to mediapipe detector 
detector = vision.ObjectDetector.create_from_options(voptions)

# Reference to tracker
tracker = Mtracker("test", timeout=1000)

# Results count status
last_length = 0

# Init camera
cap = cv2.VideoCapture(0)

# App script ID
SCRIPT_ID = "YOUR-SCRIPT-ID"

# timeout (frames)
timeout = 4

#
queue = None


# Store image function
def store_image(frame):
    base_url = f"https://script.google.com/macros/s/{SCRIPT_ID}/exec"

    # Encode frame
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Timestamp
    fecha_hora = time.strftime('%d-%m-%Y-%H-%M-%S')

    # Parameters
    data = {
        'folder': "detecciones_camara_trampa",
        'imageName': f"captura-{fecha_hora}.jpg",
        'imageType': 'image/jpeg',
        'imageBase64': image_base64
    }

    # Send request
    try:
        response = requests.post(base_url, data=data, timeout=10)
        print('Respuesta:', response.text)
    except Exception as e:
        print('Error al enviar imagen:', e)

# Thread function
def store_image_async(frame):
    thread = threading.Thread(target=store_image, args=(frame,))
    thread.start()
    
    
# Main loop
while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            break

        # Create rgb image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MPImage object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        iw = mp_image.width
        ih = mp_image.height

        # Request detections
        detection_result = detector.detect(mp_image)
        
        # Build results
        results = [
            {
                "bbox": [
                    # By default mediapipe gives you pixel coordinates. Normalizing when required
                    bbx.origin_x / iw,
                    bbx.origin_y / ih,
                    (bbx.origin_x + bbx.width) / iw,
                    (bbx.origin_y + bbx.height) / ih
                ],
                "centroid": [
                    iw / (bbx.origin_x + bbx.width),
                    ih / (bbx.origin_y + bbx.height),
                ],
                "class_name": det.categories[0].category_name,
                "class_id": yolo_cls[det.categories[0].category_name],
                "score": det.categories[0].score
            }
            for det in detection_result.detections for bbx in [det.bounding_box]
        ]
        
        # Filter persons and animals only
        results = [res for res in results if res["class_id"] in [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]]
        
        # Update tracks
        results = tracker.update(results, time.time())
        
        # Draw bbox and label for each result
        for result in results:
            # Retrieve required data
            bbx = result["bbox"]
            name = result["class_name"]
            oid = result["id"]
            x1 = int(bbx[0] * iw)
            y1 = int(bbx[1] * ih)
            x2 = int(bbx[2] * iw)
            y2 = int(bbx[3] * ih)
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # Draw label
            cv2.putText(frame, f'{name}: {oid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Validate and store image
        if len(results) != last_length:
            # New object detected
            if len(results) > last_length:
                queue = 0
            # Update current length
            last_length = len(results)
                
        # Draw results
        cv2.imshow('Object Detection', frame)
        
        if queue is not None:
            queue += 1
            if queue > timeout:
                store_image_async(frame.copy())
                # reset queue
                queue = None
            
        # Press esc to leave program
        if cv2.waitKey(1) & 0xFF == 27:
            break
    except Exception as err:
        print(f"Pipeline error: {err}")

# Release hardware and software resources
cap.release()
cv2.destroyAllWindows()

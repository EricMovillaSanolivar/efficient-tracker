import json
import time
import cv2
import torch
import numpy as np
from flask_cors import CORS
from ultralytics import YOLO
from flask import Flask, request, jsonify
from mtracker import Mtracker


classes = {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush"
}

# Check CUDA availability
cudaAvailable = torch.cuda.is_available()

# Initialize YOLOv11 model on the appropriate device
device = 'cuda' if cudaAvailable else 'cpu'
model = YOLO('yolo11s.pt').to(device)
print(f"Model loaded on {device}")

# Flask app instance
app = Flask(__name__)
# Enable cors origin for all paths
CORS(app)

tracker = Mtracker("test-only", timeout=1500)


@app.route("/fences", methods=["POST"])
def fences():
    return jsonify({ "presets": []}), 200

# Endpoint for detections (http://localhost:5000/detect)
@app.route('/detect', methods=['POST'])
def detect():
    """
    Args:
        image (binary): Blob image
        normalized (bool): Want normalized result
    Returns:
        results (json):
            - tracks (list):
                - bbox (list): [x1, y1, x2, y2]
                - centroid (list): [x_center, y_center]
                - class_name (str): "object_category_name"
                - class_id (int): yolo_class_number
                - score (float): detection_confidence
                - id (int): tracking_id
    """
    global classes
    try:
        global prev_shape
        # 1. Validate params
        if 'image' not in request.files:
            print("Error: image param not received.")
            return jsonify({'error': 'Missing parameters: "image" (file) is required.', "tracks": []}), 200

        # 2. Retrieve params
        # Retrieve and convert image
        file = request.files['image']
        image_bytes = file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        # Retrieve normalized param
        normalized = request.form.get("normalized", "false").lower() == "true"
        
        # 3. Validate image processing
        if frame is None:
            print("Error: Failed to process image file.")
            return jsonify({'error': 'Failed to process the image.', "tracks": []}), 200
        # Get image dimensions
        ih, iw = frame.shape[:2]

        # 4. Detection
        results = model(frame)
        
        # 5. Build results
        # Validation
        if len(results[0].boxes.cls.tolist()) == 0:
            return jsonify({ "tracks": []}), 200
        # Build results
        tracks = [
                {
                    "bbox": [
                        (x1 / iw if normalized else x1),
                        (y1 / ih if normalized else y1),
                        (x2 / iw if normalized else x2),
                        (y2 / ih if normalized else y2)
                    ],
                    "centroid": [
                        ((x1 + x2) / 2) / iw if normalized else (x1 + x2) / 2,
                        ((y1 + y2) / 2) / ih if normalized else (y1 + y2) / 2
                    ],
                    "class_name": classes[str(int(cls.item()))],
                    "class_id": int(cls.item()),
                    "score": float(conf.item()),
                    "id": 0
                }
                for xyxy, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf)
                for (x1, y1, x2, y2) in [xyxy.tolist()]
            ]
        
        # Request tracks
        tracks = tracker.update(tracks, time.time())
        
        # Return results
        print(f"Detection succesfully with {len(tracks)} results")
        return jsonify({'tracks': tracks}), 200

    except Exception as err:
        print(f'Error processing the request: {err}')
        return jsonify({"error": str(err),'tracks': []}), 500
    

# Exec flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9656, debug=True)

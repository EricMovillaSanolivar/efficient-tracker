# 
# IMPORTANT:
# Before to use this script, set an environtment variable on your os as "TRAP_CAMERA_APPSCRIPT"
# with your appscript id, that receives and store the image.
# 
print("Initializing")
try:
    import os
    import re
    import cv2
    import time
    import json
    import signal
    import base64
    import argparse
    import requests
    import threading
    import mediapipe as mp
    from mtracker import Mtracker
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as err:
    print(f"Error loading: {err}")

# Reference to tracker
project_id = "trap-cam1"
tracker = Mtracker(timeout=1.5)

# Flags for interface
parser = argparse.ArgumentParser()
parser.add_argument("--enable-gui", action="store_true", help="Run using GUI interface")
args = parser.parse_args()

# Trap cam engine is runing
running = True
# Track id history
history = []
new_class = None
# Results count status
last_length = 0
# timeout (frames)
timeout = 3
# Pic queue
queue = {}
local_folder = "./detecciones_locales"
os.makedirs(local_folder, exist_ok=True)
RETRY_INTERVAL = 1800  # 30 minutes
last_retry = time.time()

# Handle close
def handle_exit(signum, frame):
    global running
    print(f"Received signal -> {signum}. Exiting...")
    running = False
# Capture Ctrl+C (SIGINT) and kill (SIGTERM)
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

# Attempt to load pi camera
try:
    from picamera2 import Picamera2
    # Init camera
    cap = Picamera2()
    # Configura el modo de preview
    cap.preview_configuration.main.size = (1280, 720)
    cap.preview_configuration.main.format = "RGB888"
    cap.configure("preview")
    cap.start()
    is_picam = True
    print("Picamera loaded succesfully")
# Try to load system default camera
except Exception as err:
    print("Picamera not available, attempting to load default camera")
    cap = cv2.VideoCapture(0)
    is_picam = False
    if not cap.isOpened():
        raise IOError("Can't access to default system camera.")
    print("System default camera loaded succesfully")


# Load yolo classes equivalent
yolo_cls = None
try:
    with open("./classes.json", "r") as file:
        yolo_cls = json.load(file)
    print("Classes.json succesfully loaded")
except Exception as err:
    print(f"Error while trying to load classes.json {err}")
    raise SystemExit

# Validate gui
has_gui = False

if args.enable_gui:
    if os.environ.get("DISPLAY"):
        try:
            cv2.namedWindow("TrapCam", cv2.WINDOW_NORMAL)
            has_gui = True
            print("Host has GUI")
        except cv2.error:
            print("Host has not GUI (cv2 error)")
    else:
        print("DISPLAY not set, running headless")

# Define model path
vmodel_path = python.BaseOptions(model_asset_path='./models/efficientdet_lite2.tflite', delegate=python.BaseOptions.Delegate.CPU)
# Define model options
voptions = vision.ObjectDetectorOptions(base_options=vmodel_path, score_threshold=0.3, max_results=100)
# Reference to mediapipe detector 
detector = vision.ObjectDetector.create_from_options(voptions)
print("Mediapipe model succesfully loaded")

# App script ID
SCRIPT_ID = os.getenv("TRAP_CAMERA_APPSCRIPT")
script_failed = SCRIPT_ID is None
print(f"Script id: {SCRIPT_ID}")

# Store image function
def store_image(frame, className="Unknown", isStored=False):
    if script_failed:
        print("Theres no script id to execute")
        return False
    
    print("Attempting to save file on cloud")
    global local_folder
    # Build URL
    base_url = f"https://script.google.com/macros/s/{SCRIPT_ID}/exec"

    # Encode frame
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Timestamp
    fecha_hora = time.strftime('%d-%m-%Y_%H%M%S')

    # Parameters
    data = {
        'folder': "detecciones_camara_trampa",
        'imageName': f"{className}-{fecha_hora}.jpg",
        'imageType': 'image/jpeg',
        'imageBase64': image_base64
    }

    # Send request
    try:
        response = requests.post(base_url, data=data, timeout=10)
        js = response.json()
        print(str(js))
        if "error" in js:
            raise ValueError(f"Error reportado por el servidor: {js['error']}")
        print("File saved succesfully...")
        return True
    except Exception as e:
        print('Error al enviar imagen, guardando de manera local. error:', e)
        # Store locally
        local_path = os.path.join(local_folder, f"{className}-{fecha_hora}.jpg")
        try:
            if not isStored:
                with open(local_path, "wb") as f:
                    f.write(buffer)
                print(f"Imagen guardada localmente en: {local_path}")
        except Exception as err:
            raise ValueError(f"No se pudo guardar localmente la imagen: {err}")
        return False
    
# load local stored images and atempt to save it
def retry_stored_images():
    global local_folder

    # Verify directory and content
    if not os.path.exists(local_folder):
        print(f"Directory {local_folder} doesn't exists.")
        return

    # Search files
    files = [f for f in os.listdir(local_folder) if f.endswith(".jpg")]

    if not files:
        print("There are no images to upload.")
        return

    print(f"Loading {len(files)} images.")

    for fl in files:
        local_path = os.path.join(local_folder, fl)

        # Extract className from filename (formato: className-fecha.jpg)
        match = re.match(r"(.+)-\d{2}-\d{2}-\d{4}_\d{6}\.jpg", fl)
        if match:
            class_name = match.group(1)
        else:
            print(f"Class not found in file name: {fl}")
            continue

        # Read image
        frame = cv2.imread(local_path)
        if frame is None:
            print(f"Can't read the file: {fl}")
            continue

        # Attempt to upload image again
        try:
            loaded = store_image(frame, className=class_name, isStored=True)
            # Validate
            if loaded:
                os.remove(local_path)
                print(f"Imagen {fl} subida y eliminada localmente.")
            else:
                print("Something went wrong while trying to upload the file")
        except Exception as e:
            raise ValueError(f"Error trying to upload file {fl}: {e}")

        
print("Initializing...")
# Main loop
while running:
    try:
        if not is_picam:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = cap.capture_array()
            if frame is None:
                break

        # Create rgb image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a MPImage object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # mp_image = mp.Image(image_format=mp.ImageFormat.GRAY8, data=rgb_frame)
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
        results = tracker.update(project_id, results, time.time())
        
        # Filter results based on history
        results = [res for res in results if res["id"] not in history]
        
        # Draw bbox and label for each result
        for result in results:
            
            bbx = result["bbox"]            
            x1 = int(bbx[0] * iw)
            y1 = int(bbx[1] * ih)
            x2 = int(bbx[2] * iw)
            y2 = int(bbx[3] * ih)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            # Delay a frame to avoid flickering
            if result["id"] not in queue:
                # Add to queue
                queue[result["id"]] = timeout
                continue
            else:
                # Decrease timeout
                queue[result["id"]] -= 1
                if queue[result["id"]] > 0:
                    continue
            # Copy frame
            frm = frame.copy()
            # New object detected, store image
            history.append(result["id"])
            # Retrieve required data
            name = result["class_name"]
            oid = result["id"]
            
            # Draw rectangle
            cv2.rectangle(frm, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # Draw label
            cv2.putText(frm, f'{name}: {oid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'{name}: {oid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            threading.Thread(target=store_image, args=(frm.copy(), result["class_name"]), daemon=False).start()
            # Remove from queue
            del queue[result["id"]]
        
        # Draw results
        if has_gui:
            cv2.imshow('TrapCam', frame)
            # Press esc to leave program
            if cv2.waitKey(1) & 0xFF == 27:
                running = False
        # Validate last offline attempt
        if time.time() - last_retry >= RETRY_INTERVAL:
            threading.Thread(target=retry_stored_images, daemon=True).start()
            last_retry = time.time()
            
    except Exception as err:
        threading.Thread(target=retry_stored_images, daemon=True).start()
        print(f"Pipeline error: {err}")
        

# Release hardware and software resources
if is_picam:
    cap.stop()
else:
    cap.release()
if has_gui:
    cv2.destroyAllWindows()

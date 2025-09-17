# 
# IMPORTANT:
# Before to use this script, set an environtment variable on your os as "TRAP_CAMERA_APPSCRIPT"
# with your appscript id, that receives and store the image.
# 
print("Initializing script...")
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
    import platform
    import threading
    import mediapipe as mp
    from mtracker import Mtracker
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as err:
    print(f"Error loading: {err}")
    

# Reference to tracker
project_id = "finca"
tracker = Mtracker(timeout=4)

# Flags for interface
parser = argparse.ArgumentParser()
parser.add_argument("--enable-gui", action="store_true", help="Run using GUI interface")
args = parser.parse_args()

# Trap cam engine is runing
running = True
# Track id history
# history = []
history = { "cam1": [], "cam2": [] }
new_class = None
# Results count status
last_length = 0
# timeout (frames)
timeout = 3
# Pic queue
queue = { "cam1": {}, "cam2": {} }
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

# Attempt to load cameras
cams = {}
is_picam= False

try:
    from picamera2 import Picamera2

    # Primera cámara (puerto 0)
    cam1 = Picamera2(camera_num=0)
    cfg1 = cam1.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        sensor={"output_size": (2304, 1296)}
    )
    cam1.configure(cfg1)
    cam1.start()

    # Segunda cámara (puerto 1)
    cam2 = Picamera2(camera_num=1)
    cfg2 = cam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"},
        sensor={"output_size": (2304, 1296)}
    )
    cam2.configure(cfg2)
    cam2.start()

    cams["cam1"] = cam1
    cams["cam2"] = cam2
    is_picam = True
    print("✔ Ambas cámaras Picamera2 cargadas correctamente")
# Try to load system default camera
except Exception as err:
    print("Picamera not available, attempting to load default camera")
    try:
        cam1 = cv2.VideoCapture(0)
        if not cam1.isOpened():
            raise IOError("❌ No se pudo abrir la cámara USB en /dev/video0")
        
        cam2 = cv2.VideoCapture(1)
        if not cam2.isOpened():
            raise IOError("❌ No se pudo abrir la cámara USB en /dev/video1")

        cams["cam1"] = cam1
        cams["cam2"] = cam2

        print("✔ Dos cámaras USB cargadas correctamente")
    except Exception as err:
        raise RuntimeError(f"❌ Error inicializando cámaras USB: {err}")


# Load yolo classes equivalent
yolo_cls = None
try:
    with open("./classes.json", "r") as file:
        yolo_cls = json.load(file)
    print("Classes.json succesfully loaded")
except Exception as err:
    print(f"Error while trying to load classes.json {err}")
    raise SystemExit

# Define model path
vmodel_path = python.BaseOptions(model_asset_path='./models/efficientdet_lite2.tflite', delegate=python.BaseOptions.Delegate.CPU)
# Define model options
voptions = vision.ObjectDetectorOptions(base_options=vmodel_path, score_threshold=0.3, max_results=100)
# Reference to mediapipe detector 
models = [
    vision.ObjectDetector.create_from_options(voptions),
    vision.ObjectDetector.create_from_options(voptions)
]
print("Mediapipe model succesfully loaded")

# App script ID
SCRIPT_ID = os.getenv("TRAP_CAMERA_APPSCRIPT")
script_failed = SCRIPT_ID is None
print(f"Script id: {SCRIPT_ID}")

# Store image function
def store_image(annotated, source, frame, className="Unknown"):
    global local_folder
    # Validate script is working
    if script_failed:
        print("Theres no script id to execute")
        return False
    
    # Create date
    date_time = time.strftime('%d-%m-%Y_%H%M%S')
        
    print("Attempting to save file on cloud")
    # Build URL
    base_url = f"https://script.google.com/macros/s/{SCRIPT_ID}/exec"

    # Encode frame
    _, buffer = cv2.imencode('.jpg', frame)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    file_name = f"{project_id}-{source}-{className}{'-box' if annotated else ''}-{date_time}.jpg"

    # Parameters
    data = {
        'folder': "detecciones_camara_trampa",
        'imageName': file_name,
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
    # If an error, store locally
    except Exception as e:
        print('Error al enviar imagen, guardando de manera local. error:', e)
        # Store locally
        local_path = os.path.join(local_folder, file_name)
        try:
            with open(local_path, "wb") as f:
                f.write(buffer)
            print(f"Imagen guardada localmente en: {local_path}")
        except Exception as err:
            raise ValueError(f"No se pudo guardar localmente la imagen: {err}")
        return False
    
# load local stored images and atempt to save it
def retry_stored_images():
    global local_folder
    if script_failed:
        print("⚠ No hay SCRIPT_ID configurado, no se pueden reintentar subidas.")
        return

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

        # Read image
        frame = cv2.imread(local_path)
        if frame is None:
            print(f"⚠ No se pudo leer el archivo: {fl}")
            continue

        # Base64 encode
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Build request
        base_url = f"https://script.google.com/macros/s/{SCRIPT_ID}/exec"
        data = {
            'folder': "detecciones_camara_trampa",
            'imageName': fl,
            'imageType': 'image/jpeg',
            'imageBase64': image_base64
        }

        try:
            response = requests.post(base_url, data=data, timeout=10)
            js = response.json()
            if "error" in js:
                print(js["error"])
                continue
            print(f"✔ Imagen subida: {fl}")
            os.remove(local_path)
        except Exception as e:
            print(f"❌ Error al subir {fl}: {e}")
            

# Validate gui
has_gui = False

if args.enable_gui:
    system = platform.system()
    if system == "Linux":
        if os.environ.get("DISPLAY"):
            try:
                cv2.namedWindow("TrapCam", cv2.WINDOW_NORMAL)
                has_gui = True
                print("Host has GUI (Linux with DISPLAY)")
            except cv2.error:
                print("Host has not GUI (cv2 error)")
        else:
            print("DISPLAY not set, running headless")
    else:
        # En Windows y Mac no se necesita DISPLAY
        try:
            cv2.namedWindow("TrapCam", cv2.WINDOW_NORMAL)
            has_gui = True
            print(f"Host has GUI ({system})")
        except cv2.error:
            print(f"Host has not GUI on {system} (cv2 error)")
        
        
print("Initializing main loop...")
retry_stored_images()
# Main loop
while running:
    try:
        # start time
        strt = time.time()
        frames_to_show = {}
        curr_det = 0
        for cam_name, cam in cams.items():
                if not is_picam:
                    ret, frame = cam.read()
                    if not ret:
                        break
                else:
                    frame = cam.capture_array()
                    if frame is None:
                        break
                
                clean_frame = frame.copy()
                
                if curr_det == 0:
                    # Create rgb image
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    # 🔹 Convert frame to grayscale (1 channel)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # 🔹 Expand grayscale to 3 channels so it's still compatible with RGB models
                    rgb_frame = cv2.merge([gray, gray, gray])

                # Create a MPImage object
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                # mp_image = mp.Image(image_format=mp.ImageFormat.GRAY8, data=rgb_frame)
                iw = mp_image.width
                ih = mp_image.height

                # Request detections
                detection_result = models[curr_det].detect(mp_image)
                curr_det += 1
                
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
                results = [res for res in results if res["class_id"] in [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] and res["score"] > 0.4]
                
                # Update tracks
                results = tracker.update(f"{project_id}-{cam_name}", results, time.time())
                
                # Filter results based on history
                results = [res for res in results if res["id"] not in history[cam_name]]
                
                # Draw bbox and label for each result
                for result in results:
                    
                    bbx = result["bbox"]            
                    x1 = int(bbx[0] * iw)
                    y1 = int(bbx[1] * ih)
                    x2 = int(bbx[2] * iw)
                    y2 = int(bbx[3] * ih)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    
                    # Delay a frame to avoid flickering
                    if result["id"] not in queue[cam_name]:
                        queue[cam_name][result["id"]] = timeout
                        continue
                    else:
                        queue[cam_name][result["id"]] -= 1
                        if queue[cam_name][result["id"]] > 0:
                            continue
                    
                    # New object detected, store image
                    history[cam_name].append(result["id"])
                    # Retrieve required data
                    name = result["class_name"]
                    oid = result["id"]
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    # Draw label
                    cv2.putText(frame, f'{name}: {oid}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Copy frame
                    frm = frame.copy()
                    # Store clean frame
                    threading.Thread(target=store_image, args=(False, cam_name, clean_frame, result["class_name"]), daemon=False).start()
                    # Store annotated
                    threading.Thread(target=store_image, args=(True, cam_name, frm, result["class_name"]), daemon=False).start()
                    # Remove from queue
                    del queue[cam_name][result["id"]]
        
        print(f"Pipeline time: {time.time() - strt}")
                
    
        if has_gui and frames_to_show:
            # Juntar horizontalmente si ambas están disponibles
            if len(frames_to_show) == 2:
                cam1_frame = frames_to_show.get("cam1")
                cam2_frame = frames_to_show.get("cam2")

                # Redimensionar si no son iguales
                if cam1_frame.shape != cam2_frame.shape:
                    h = min(cam1_frame.shape[0], cam2_frame.shape[0])
                    w = min(cam1_frame.shape[1], cam2_frame.shape[1])
                    cam1_frame = cv2.resize(cam1_frame, (w, h))
                    cam2_frame = cv2.resize(cam2_frame, (w, h))

                combined = cv2.hconcat([cam1_frame, cam2_frame])
            else:
                # Solo una cámara disponible
                combined = list(frames_to_show.values())[0]

            cv2.imshow('TrapCam', combined)

            # Salir con ESC
            if cv2.waitKey(1) & 0xFF == 27:
                running = False
                
    except Exception as err:
        threading.Thread(target=retry_stored_images, daemon=True).start()
        print(f"Pipeline error: {err}")
        

# Release hardware and software resources
for cam_name, cam in cams.items():
    if is_picam:
        cam.stop()
    else:
        cam.release()
if has_gui:
    cv2.destroyAllWindows()

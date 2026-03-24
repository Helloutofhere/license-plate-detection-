import cv2
import re
import pandas as pd
import easyocr
from ultralytics import YOLO
from datetime import datetime
import os

# --- Configuration & System Requirements [cite: 47-70] ---
MODEL_PATH = 'models/best.pt'  # Path to your trained YOLOv8 weights [cite: 35, 84]
VIDEO_SOURCE = 'input/traffic_video.mp4'  # Or 0 for live webcam [cite: 58]
OUTPUT_CSV = 'output/vehicle_logs.csv'

# Initialize Models
model = YOLO(MODEL_PATH) 
reader = easyocr.Reader(['en'], gpu=True) # Uses GPU if available [cite: 53, 65]
data_log = []

def apply_preprocessing(plate_img):
    """
    Step 2 & 4: Image preprocessing to improve OCR accuracy.
    Handles issues like noise and low contrast[cite: 130, 137].
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) [cite: 26, 82]
    # Bilateral filter removes noise while keeping edges sharp [cite: 40, 91]
    processed = cv2.bilateralFilter(gray, 11, 17, 17)
    return processed

def validate_indian_plate(text):
    """
    Post-Processing: Cleans OCR noise and validates against Indian formats[cite: 92, 93].
    Addresses common OCR confusion (e.g., 0 vs O)[cite: 136].
    """
    # Remove non-alphanumeric characters [cite: 92]
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Regex for Indian Format: State(2) + District(1-2) + Type(1-2) + Number(4) [cite: 33, 127]
    # Examples validated: AP10AR0658, DL7CD5017 [cite: 101, 104]
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
    
    if re.match(pattern, clean_text):
        return clean_text
    return None

def run_anpr():
    # Ensure output directory exists [cite: 57]
    if not os.path.exists('output'):
        os.makedirs('output')

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    print("Starting ANPR System...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 3: Number Plate Detection using YOLOv8 [cite: 24, 83]
        results = model.predict(frame, conf=0.5, verbose=False) [cite: 36, 85]
        
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates [cite: 36, 72]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2] # Crop plate region [cite: 22, 37, 86]

                # Step 4: OCR with Preprocessing [cite: 23, 38, 88]
                processed_plate = apply_preprocessing(plate_crop)
                ocr_results = reader.readtext(processed_plate) [cite: 39, 90]

                for (bbox, text, prob) in ocr_results:
                    valid_plate = validate_indian_plate(text)
                    
                    # Filter by confidence and format validation [cite: 44, 96]
                    if valid_plate and prob > 0.4:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Step 5: Log to Database (CSV) 
                        if not any(d['Plate'] == valid_plate for d in data_log[-5:]): # Simple de-duplication
                            data_log.append({"Time": timestamp, "Plate": valid_plate})
                            print(f"Detected: {valid_plate} at {timestamp}")

                        # Visualization: Draw box and text on video feed [cite: 94, 102]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, valid_plate, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the real-time feed [cite: 45, 46, 94]
        cv2.imshow('Automatic Number Plate Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save final logs to CSV [cite: 95]
    if data_log:
        df = pd.DataFrame(data_log)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Results saved to {OUTPUT_CSV}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_anpr()

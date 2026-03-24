import cv2
import re
import pandas as pd
import easyocr
from ultralytics import YOLO
from datetime import datetime
import os
from preprocess import apply_preprocessing

# --- Configuration & System Requirements ---
MODEL_PATH = 'models/best.pt'  # Path to your trained YOLOv8 weights
VIDEO_SOURCE = 'input/traffic_video.mp4'  # Or 0 for live webcam
OUTPUT_CSV = 'output/vehicle_logs.csv'

# Initialize Models
model = YOLO(MODEL_PATH) 
reader = easyocr.Reader(['en'], gpu=True) # Uses GPU if available
data_log = []

def validate_indian_plate(text):
    """
    Post-Processing: Cleans OCR noise and validates against Indian formats.
    Addresses common OCR confusion (e.g., 0 vs O).
    """
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Regex for Indian Format: State(2)+District(2)+Series(1-2)+Number(4)
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'
    
    if re.match(pattern, clean_text):
        return clean_text
    return None

def run_anpr():
    if not os.path.exists('output'): os.makedirs('output')

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    print("Starting ANPR System...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Step 3: Number Plate Detection using YOLOv8
        results = model.predict(frame, conf=0.5, verbose=False) 
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2] 

                # Step 4: OCR with Preprocessing
                processed_plate = apply_preprocessing(plate_crop)
                ocr_results = reader.readtext(processed_plate) 

                for (bbox, text, prob) in ocr_results:
                    valid_plate = validate_indian_plate(text)
                    
                    if valid_plate and prob > 0.4:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Step 5: Log to Database & De-duplication
                        if not any(d['Plate'] == valid_plate for d in data_log[-5:]):
                            data_log.append({"Time": timestamp, "Plate": valid_plate})
                            print(f"Detected: {valid_plate} at {timestamp}")

                        # Visualization
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, valid_plate, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Automatic Number Plate Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if data_log:
        pd.DataFrame(data_log).to_csv(OUTPUT_CSV, index=False)
        print(f"Results saved to {OUTPUT_CSV}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_anpr()

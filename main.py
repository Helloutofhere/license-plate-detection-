import cv2
import pandas as pd
import easyocr
from ultralytics import YOLO
from datetime import datetime

# 1. System Initialization [cite: 41, 64, 65]
model = YOLO('models/best.pt')  # Your trained YOLOv8 model [cite: 35]
reader = easyocr.Reader(['en']) # English OCR [cite: 38]
data_log = []

def apply_preprocessing(plate_img):
    """Refined preprocessing based on Step 2 & 4 of Methodology """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Noise reduction & Contrast enhancement [cite: 40, 91]
    distorted = cv2.GaussianBlur(gray, (5, 5), 0)
    return distorted

def start_anpr(source=0):
    cap = cv2.VideoCapture(source) # Supports camera or video file [cite: 58]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Step 3: Detection Phase [cite: 83, 85]
        results = model.predict(frame, conf=0.5) 
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2] 

                # Step 4: OCR Phase [cite: 88, 89]
                processed_plate = apply_preprocessing(plate_crop)
                ocr_data = reader.readtext(processed_plate)

                for (bbox, text, prob) in ocr_data:
                    if prob > 0.5: # Filters out low-confidence reads [cite: 44]
                        plate_text = text.upper().replace(" ", "")
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Step 5: Post-Processing & Output [cite: 93, 95]
                        data_log.append({"Time": timestamp, "Plate": plate_text})
                        
                        # Real-time Annotation [cite: 94]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, plate_text, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('ANPR Real-Time System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Save to Database (CSV) 
    if data_log:
        pd.DataFrame(data_log).to_csv("output/vehicle_logs.csv", index=False)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_anpr('input_video.mp4')

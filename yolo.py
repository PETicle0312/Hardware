from ultralytics import YOLO
import cv2
import numpy as np
import serial
import time
import requests
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

ser = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)

model = YOLO('C:/Users/jungs/Desktop/hywu/3_grade/PETicle/train2/weights/best.pt')


BACKEND_API_URL = 'http://192.168.183.1:8080/api/device/input'
DEVICE_ID = 12345



cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

studentNumber = input("학번 입력 : ")

count = 0


def send_data_to_backend(student_number, device_id, input_count):
    payload = {
        'studentNumber': student_number,
        'deviceId': device_id,
        'inputCount': input_count,
        'inputTime': datetime.now().isoformat()
    }
    logger.info("Backend input transmission started.")
    try:
        response = requests.post(BACKEND_API_URL, json=payload, timeout=5)
        response.raise_for_status()
        logger.info("Backend input transmission succeeded.")
        return True
    except requests.exceptions.Timeout:
        logger.warning("Backend input transmission timed out.")
        return False
    except requests.exceptions.RequestException:
        logger.exception("Backend input transmission failed.")
        return False

while True:
            ret, frame = cap.read()
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("Result", frame)
            key = cv2.waitKey(10)
            if key == 27:  # ESC 키
                logger.info("Bottle input session finished. inputCount=%s", count)
                if count > 0:
                    send_data_to_backend(studentNumber, DEVICE_ID, count)
                break
            elif key != ord(' ') :
                continue

            results = model.predict(frame, task="classify", verbose=False)

            r = results[0]
            top1_index = r.probs.top1 
            top1_conf = r.probs.top1conf
            class_name = r.names[top1_index]

            logger.debug("Bottle classification completed. class=%s confidence=%.2f", class_name, top1_conf)

            if class_name == 'background':
                time.sleep(1.0)

            elif class_name == 'clean':
                if top1_conf > 0.95 :
                    logger.debug("Clean bottle detected.")
                    ser.write(b'L')
                    count += 1
                else :
                    logger.debug("Bottle classification confidence was below the acceptance threshold.")
                    ser.write(b'R')
                time.sleep(2.0)
                
            elif class_name =='label':
                logger.debug("Labelled bottle detected.")
                ser.write(b'R')
                time.sleep(2.0)

            else :
                logger.debug("Non-PET item detected.")
                ser.write(b'R')
                time.sleep(2.0)


cap.release()
cv2.destroyAllWindows()
ser.close()

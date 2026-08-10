from ultralytics import YOLO
import cv2
import numpy as np
import serial
import time
import requests
from datetime import datetime

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
    print(f"Attempting to send data to backend: {payload}")
    try:
        response = requests.post(BACKEND_API_URL, json=payload, timeout=5)
        response.raise_for_status()
        try:
            response_data = response.json()
        except ValueError:
            response_data = response.text
        print(f"Data sent successfully! Server response: {response_data}")
        return True
    except requests.exceptions.Timeout:
        print("Error: Request to backend timed out.")
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to backend: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server error response: {e.response.text}")
        return False

while True:
            ret, frame = cap.read()
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("Result", frame)
            key = cv2.waitKey(10)
            if key == 27:  # ESC 키
                print(f"{studentNumber} 학생은 총 {count}개의 PET을 투입했습니다.")
                if count > 0:
                    send_data_to_backend(studentNumber, DEVICE_ID, count)
                break
            elif key != ord(' ') :
                continue

            results = model.predict(frame, task="classify", verbose=False)

            r = results[0]
            print(r.probs)
            top1_index = r.probs.top1 
            top1_conf = r.probs.top1conf
            class_name = r.names[top1_index]

            print(f"Predicted class: {class_name} ({top1_conf:.2f})")

            if class_name == 'background':
                time.sleep(1.0)

            elif class_name == 'clean':
                if top1_conf > 0.95 :
                    print("정상")
                    ser.write(b'L')
                    count += 1
                else :
                    print("이물질이 있을 수 있음.")
                    ser.write(b'R')
                time.sleep(2.0)
                
            elif class_name =='label':
                print("라벨있음")
                ser.write(b'R')
                time.sleep(2.0)

            else :
                print("패트병이 아님")
                ser.write(b'R')
                time.sleep(2.0)


cap.release()
cv2.destroyAllWindows()
ser.close()

from ultralytics import YOLO
import cv2
import numpy as np
import serial
import time
import requests # 이 줄 추가
from datetime import datetime # 이 줄 추가

# 시리얼 포트 연결
ser = serial.Serial('COM5', 9600, timeout=1)
time.sleep(2)

# YOLOv8 Classification 모델 로드
model = YOLO('C:/Users/jungs/Desktop/hywu/3_grade/PETicle/train2/weights/best.pt')


#백엔드 연동을 위한 설정
BACKEND_API_URL = 'http://192.168.183.1:8080/api/device/input' # <-- 이 부분을 여러분의 Spring 서버 주소로 변경!
DEVICE_ID = 12345 # <-- 이 자판기의 고유 ID를 숫자로 설정하세요



# 카메라 연결
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# 움직임 감지 변수 초기화
#prev_frame = None
#last_detect_time = 0
#COOLDOWN = 2  # 분류 후 대기시간
#DELAY = 1     # 감지 후 처리 지연

count = 0
studentNumber = input("학번 입력 : ")

def send_data_to_backend(student_number, device_id, input_count):
    """
    백엔드 서버로 페트병 데이터를 전송하는 함수
    """
    payload = {
        #'userId': user_id,
        'studentNumber': student_number, # DTO에 studentNumber 필드가 있다면 userId와 동일하게 설정 (학번이 userId인 경우)
        'deviceId': device_id,
        'inputCount': input_count,
        'inputTime': datetime.now().isoformat() # ISO 8601 형식으로 현재 시간 전송
    }
    print(f"Attempting to send data to backend: {payload}")
    try:
        response = requests.post(BACKEND_API_URL, json=payload, timeout=5) # 5초 타임아웃
        response.raise_for_status() # HTTP 에러(4xx, 5xx) 발생 시 예외 처리
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
            #print ("Image Capturing")
            ret, frame = cap.read()
            ret, frame = cap.read()

            if not ret:
                break

            cv2.imshow("Result", frame)
            key = cv2.waitKey(10)
            if key == 27:  # ESC 키
                print(studentNumber, " 학생은 ", count, "개의 PET을 투입했습니다.")
                break
            elif key != ord(' ') :
                continue

    # YOLO 분류 실행
            results = model.predict(frame, task="classify", verbose=False)

    #for r in results:
            r = results[0]
            print(r.probs)
            # top1 class 가장 높은 신뢰도로 분류된 결과
            top1_index = r.probs.top1 
            top1_conf = r.probs.top1conf
            class_name = r.names[top1_index]

            print(f"Predicted class: {class_name} ({top1_conf:.2f})") #예측 결과 출력력
            #f-string : 문자열 포맷팅

            # 조건별 모터 조작
            if class_name == 'background':
                #print("배경")
                time.sleep(1.0)
                #continue  # skip

            elif class_name == 'clean':
                if top1_conf > 0.95 :
                    print("정상")
                    ser.write(b'L')  # 모터 왼쪽 회전
                    count = count + 1
                    #point 저장 로직 또는 API 호출 위치 (svc call or DB update)
                    send_data_to_backend(studentNumber, DEVICE_ID, 1) # 함수 호출
                else :
                    print("이물질이 있을 수 있음.")
                    ser.write(b'R')  # 모터 왼쪽 회전
                time.sleep(2.0)
                
            elif class_name =='label':
                print("라벨있음")
                ser.write(b'R')  # 모터 오른쪽 회전
                time.sleep(2.0)

            else :
                print("패트병이 아님")
                ser.write(b'R')  # 모터 오른쪽 회전
                time.sleep(2.0)


cap.release()
cv2.destroyAllWindows()
ser.close()

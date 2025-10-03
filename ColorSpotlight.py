import cv2
import mediapipe as mp
import numpy as np
import math

# --- 1. MediaPipe 및 웹캠 설정 ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)

# --- 2. 메인 루프 ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # 최종적으로 보여줄 이미지 (초기값은 원본 컬러 이미지)
    output_image = image.copy()

    # MediaPipe 처리를 위해 BGR -> RGB 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    # 기본 배경은 흑백으로 준비
    # 1. 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 2. 컬러 이미지와 합성을 위해 다시 3채널 BGR로 변환
    gray_background = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # 스포트라이트 중심점: 손바닥의 중심이 될 중지 손가락의 첫번째 마디(MCP)로 설정
        center_node = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        center_x = int(center_node.x * w)
        center_y = int(center_node.y * h)

        # 스포트라이트 반지름: 엄지 끝과 새끼손가락 끝의 거리로 계산 (손을 펴는 정도)
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        distance = math.sqrt((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2)

        # 거리를 반지름으로 매핑
        # distance: 약 0.05(오므림) ~ 0.25(활짝 폄)
        # radius: 50px ~ 300px
        radius = int(np.interp(distance, [0.05, 0.25], [50, 300]))
        radius = np.clip(radius, 0, min(w, h))

        # --- 핵심 로직: 마스크를 이용한 합성 ---
        # 1. 검은색 배경의 마스크 생성
        mask = np.zeros((h, w), dtype=np.uint8)
        # 2. 마스크에 스포트라이트 영역을 흰색 원으로 그림
        cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
        # 3. 마스크를 3채널로 확장
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 4. np.where를 이용한 초고속 합성
        # 마스크가 흰색(>0)인 곳은 원본 컬러 이미지(output_image)를,
        # 검은색인 곳은 흑백 배경(gray_background)을 선택하여 최종 이미지 생성
        output_image = np.where(mask_3ch > 0, output_image, gray_background)

        # --- 시각화 ---
        # 스포트라이트 테두리 그리기
        cv2.circle(output_image, (center_x, center_y), radius, (255, 255, 0), 3)
        # 정보 텍스트 표시
        info_text = f'Radius: {radius}'
        cv2.putText(output_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # 손이 없으면 전체를 흑백으로 보여줌
        output_image = gray_background

    cv2.imshow('Color Spotlight Effect', output_image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
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

# 포털 내부의 색상(Hue)을 계속 바꾸기 위한 프레임 카운터
frame_counter = 0

# --- 2. 메인 루프 ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    frame_counter += 1
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # 최종 결과물 이미지 (기본값은 원본 영상)
    output_image = image.copy()

    # --- "다른 차원" 영상 만들기 (사이키델릭 효과) ---
    # 1. 영상을 HSV 색상 공간으로 변환 (Hue, Saturation, Value)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. 색상(Hue) 채널을 프레임 카운터에 따라 계속해서 변화시킴
    # Hue 값은 0~179 범위이므로, 180으로 나눈 나머지를 사용해 순환시킴
    hue_shift = (frame_counter * 2) % 180
    hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180

    # 3. 채도를 최대로 올려 색감을 더욱 강렬하게 만듦
    hsv_image[:, :, 1] = 255

    # 4. 다시 BGR 색상 공간으로 변환하여 "다른 차원" 영상 완성
    portal_feed = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # MediaPipe 처리
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # 포털의 중심: 엄지 끝과 검지 끝의 중간 지점
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        center_x = int((thumb_tip.x + index_tip.x) / 2 * w)
        center_y = int((thumb_tip.y + index_tip.y) / 2 * h)

        # 포털의 반지름: 엄지 끝과 검지 끝의 거리
        distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

        # 거리를 반지름으로 매핑 (거리가 가까울수록 작은 원)
        # distance: 약 0 (붙임) ~ 0.25 (최대한 폄)
        # radius: 0px ~ 300px
        radius = int(np.interp(distance, [0.02, 0.25], [10, 300]))
        radius = np.clip(radius, 0, min(w, h))

        # --- 핵심 로직: 마스크를 이용한 합성 ---
        if radius > 10:  # 아주 작은 포털은 그리지 않음
            # 1. 검은색 마스크 생성
            mask = np.zeros((h, w), dtype=np.uint8)
            # 2. 마스크에 포털 영역을 흰색 원으로 그림
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            # 3. 마스크를 3채널로 확장
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # 4. np.where를 이용한 초고속 합성
            # 마스크가 흰색인 곳은 "다른 차원"(portal_feed)을,
            # 검은색인 곳은 "현실 세계"(image)를 보여줌
            output_image = np.where(mask_3ch > 0, portal_feed, image)

            # --- 시각화 ---
            # 포털의 경계를 빛나는 효과처럼 표현
            cv2.circle(output_image, (center_x, center_y), radius, (255, 255, 0), 4)
            # 정보 텍스트 표시
            info_text = f'Portal Radius: {radius}'
            cv2.putText(output_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Chroma Key Portal', output_image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
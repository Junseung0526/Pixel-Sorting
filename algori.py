import cv2
import mediapipe as mp
import numpy as np
import math

# --- 1. MediaPipe 및 웹캠 설정 ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  # 한 손만 인식
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
cap = cv2.VideoCapture(0)


# --- 2. 픽셀 소팅 함수 정의 ---
def pixel_sort_column(column, threshold):
    """특정 세로줄(column)을 입력받아 밝기 임계값 기준으로 픽셀 소팅을 수행"""

    # 그레이스케일로 변환하여 밝기 정보 추출
    gray_column = cv2.cvtColor(column.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()

    start_index = -1

    # 각 픽셀을 순회
    for i in range(len(gray_column)):
        # 현재 픽셀의 밝기가 임계값보다 크고, 소팅이 시작되지 않았다면 시작 인덱스 기록
        if gray_column[i] > threshold and start_index == -1:
            start_index = i

        # 현재 픽셀의 밝기가 임계값보다 작고, 소팅이 진행 중이었다면 해당 구간을 소팅
        elif gray_column[i] < threshold and start_index != -1:
            # 소팅할 구간(run) 추출
            run = column[start_index:i]

            # 밝기 기준으로 정렬하기 위해 BGR -> Gray 변환 후 정렬 인덱스 획득
            run_gray_values = cv2.cvtColor(run.reshape(-1, 1, 3), cv2.COLOR_BGR2GRAY).flatten()
            sorted_indices = np.argsort(run_gray_values)

            # 정렬된 인덱스를 사용하여 원래 컬러 픽셀 정렬
            sorted_run = run[sorted_indices]

            # 정렬된 픽셀을 다시 원래 위치에 삽입
            column[start_index:i] = sorted_run

            # 시작 인덱스 초기화
            start_index = -1

    return column


# --- 3. 메인 루프 ---
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # MediaPipe 처리를 위해 BGR -> RGB 변환
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks:
        # 첫 번째로 감지된 손에 대해서만 처리
        hand_landmarks = results.multi_hand_landmarks[0]

        # 손목(WRIST)의 x좌표를 소팅할 위치로 지정
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        sort_x = int(wrist.x * w)

        # 엄지(THUMB_TIP)와 검지(INDEX_FINGER_TIP) 좌표
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # 손가락 사이의 거리 계산
        distance = math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)

        # 거리를 소팅 임계값(threshold)으로 매핑 (0~255 사이)
        # 거리가 가까울수록(최소 0.0) 임계값이 낮아지고, 멀수록(최대 약 0.3) 임계값이 높아짐
        # 거리가 0.1 이하일 때 효과가 강하게 나타나도록 매핑
        threshold = int(np.interp(distance, [0.02, 0.15], [20, 200]))
        threshold = np.clip(threshold, 0, 255)  # 0~255 범위를 벗어나지 않도록

        # 특정 제스처(핀치)일 때만 효과 활성화
        pinch_threshold = 0.05  # 이 값보다 거리가 가까우면 활성화
        if distance < pinch_threshold:
            # 소팅할 세로줄(ROI) 선택 (성능을 위해 좁은 영역만)
            start_x = max(0, sort_x - 2)
            end_x = min(w, sort_x + 2)

            if start_x < end_x:
                roi_column = image[:, start_x:end_x]

                # 각 세로줄에 대해 픽셀 소팅 적용
                for i in range(roi_column.shape[1]):
                    column_to_sort = roi_column[:, i].copy()
                    sorted_column = pixel_sort_column(column_to_sort, threshold)
                    image[:, start_x + i] = sorted_column

        # --- 시각화 ---
        # 소팅 기준선 그리기
        cv2.line(image, (sort_x, 0), (sort_x, h), (0, 255, 255), 1)
        # 손가락 거리 및 임계값 정보 표시
        info_text = f'Dist: {distance:.2f} / Thr: {threshold}'
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # 손 랜드마크 그리기
        mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Controlled Pixel Sorting', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
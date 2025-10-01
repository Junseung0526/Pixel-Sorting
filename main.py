import cv2
import mediapipe as mp
import numpy as np


def pixel_sort(image, threshold=100, direction='horizontal'):
    """
    이미지의 특정 부분을 픽셀 소팅합니다.

    Args:
        image: 픽셀 소팅을 적용할 이미지 영역 (ROI)
        threshold: 픽셀 소팅을 시작할 밝기 임계값 (0-255)
        direction: 'horizontal' 또는 'vertical'
    """
    if image is None or image.size == 0:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if direction == 'horizontal':
        # 가로 방향으로 소팅
        for y in range(image.shape[0]):
            row = image[y, :]
            gray_row = gray[y, :]

            start_index = -1
            # 임계값을 넘는 첫 픽셀 찾기
            for x in range(image.shape[1]):
                if gray_row[x] > threshold:
                    start_index = x
                    break

            if start_index != -1:
                # 소팅할 부분 추출
                sort_part = row[start_index:]
                # 밝기 값을 기준으로 픽셀 정렬
                # zip으로 픽셀과 밝기 값을 묶고, 밝기 값으로 정렬 후 다시 픽셀만 추출
                sorted_pixels = [pixel for pixel, brightness in
                                 sorted(zip(sort_part, gray_row[start_index:]), key=lambda item: item[1])]

                # 정렬된 픽셀을 다시 이미지에 적용
                image[y, start_index:] = sorted_pixels

    # (필요시 'vertical' 방향도 유사하게 구현 가능)

    return image


# --- MediaPipe 초기화 ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  # 한 개의 손만 인식하여 성능 최적화
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- 웹캠 설정 ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # 성능 향상을 위해 이미지 쓰기 불가로 설정
    image.flags.setflags(write=False)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # 손 인식
    results = hands.process(image)

    # 원본 이미지를 다시 쓰기 가능하게 하고 BGR로 변환
    image.flags.setflags(write=True)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 랜드마크 그리기 및 픽셀 소팅 적용
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. 검지손가락 끝 랜드마크(ID: 8) 좌표 얻기
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # 랜드마크 좌표는 0~1 사이의 정규화된 값이므로, 이미지 크기에 맞춰 변환
            h, w, _ = image.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # 2. 픽셀 소팅을 적용할 영역(ROI) 지정
            # 검지손가락 끝의 오른쪽 영역에 효과 적용
            roi_x_start = cx
            roi_y_start = cy - 10  # 손가락 위아래로 약간의 두께
            roi_x_end = w  # 화면 끝까지
            roi_y_end = cy + 10

            # 3. ROI가 이미지 경계를 벗어나지 않도록 처리
            roi_y_start = max(0, roi_y_start)
            roi_y_end = min(h, roi_y_end)
            roi_x_start = max(0, roi_x_start)

            if roi_y_start < roi_y_end and roi_x_start < roi_x_end:
                # 4. ROI 이미지 잘라내기
                roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

                # 5. 픽셀 소팅 함수 호출
                sorted_roi = pixel_sort(roi.copy(), threshold=120)

                # 6. 원본 이미지에 효과가 적용된 ROI 붙여넣기
                image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = sorted_roi

            # 손 관절 그리기 (효과 위에 덧그림)
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Pixel Sorting', image)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC 키 누르면 종료
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
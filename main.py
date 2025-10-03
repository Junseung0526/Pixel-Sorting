import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hands 모델 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hands 객체 생성
# - min_detection_confidence: 손 인식이 성공했다고 판단하는 최소 신뢰도
# - min_tracking_confidence: 손 추적이 성공했다고 판단하는 최소 신뢰도
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        break

    # 성능 향상을 위해 이미지를 읽기 전용으로 만들고, BGR을 RGB로 변환
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)

    # 다시 BGR로 변환하여 화면에 출력할 준비
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 손이 2개 인식되었을 때만 로직 실행
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:

        points = []
        for hand_landmarks in results.multi_hand_landmarks:
            # 랜드마크 그리기 (시각화용)
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            # 이미지의 너비와 높이
            h, w, _ = image.shape

            # 엄지 끝(THUMB_TIP, 4)과 검지 끝(INDEX_FINGER_TIP, 8)의 좌표를 리스트에 추가
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            points.append((int(thumb_tip.x * w), int(thumb_tip.y * h)))
            points.append((int(index_finger_tip.x * w), int(index_finger_tip.y * h)))

        # 4개의 점이 모두 수집되었을 경우
        if len(points) == 4:
            # np.array로 변환하여 x, y 좌표의 최소/최대값 찾기
            pts = np.array(points, dtype=np.int32)
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]

            x1, x2 = np.min(x_coords), np.max(x_coords)
            y1, y2 = np.min(y_coords), np.max(y_coords)

            # ROI 영역이 유효한지 확인 (너비와 높이가 0보다 큰지)
            if x2 > x1 and y2 > y1:
                # 1. ROI(관심 영역) 추출
                roi = image[y1:y2, x1:x2]

                # 2. ROI에 필터 적용 (예: 가우시안 블러)
                # ksize의 값은 홀수여야 함
                # blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)

                # --- 다른 필터 예시 ---
                # 흑백 필터
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred_roi = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR) # 다시 3채널로 변경해야 함

                # 캐니 엣지 필터
                # canny_roi = cv2.Canny(roi, 100, 200)
                # blurred_roi = cv2.cvtColor(canny_roi, cv2.COLOR_GRAY2BGR) # 다시 3채널로 변경해야 함
                # -------------------------

                # 3. 필터 적용된 ROI를 원본 이미지에 다시 삽입
                image[y1:y2, x1:x2] = blurred_roi

                # 생성된 사각형 테두리 그리기 (시각화용)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 결과 화면 출력
    cv2.imshow('Hand Filter Example', image)

    # ESC 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 자원 해제
hands.close()
cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

# Mediapipe 초기화
mp_face = mp.solutions.face_mesh
mp_hand = mp.solutions.hands
face = mp_face.FaceMesh(static_image_mode=True)
hands = mp_hand.Hands(static_image_mode=True, max_num_hands=2)

# 디렉토리에서 모든 PNG 파일 찾기
png_files = glob.glob("*.png")

# 결과 저장을 위한 리스트
results = []

# 그래프 설정
n_images = len(png_files)
if n_images == 0:
    print("디렉토리에 PNG 파일이 없습니다.")
    exit()

# 이미지 수에 따라 그리드 크기 조정
rows = int(np.ceil(n_images / 3))
cols = min(n_images, 3)

plt.figure(figsize=(15, 5 * rows))

for idx, img_path in enumerate(png_files):
    # 이미지 불러오기
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # 마스크 초기화
    face_mask = np.zeros((h, w), dtype=np.uint8)
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 얼굴 및 손 랜드마크 추출
    face_results = face.process(img_rgb)
    hand_results = hands.process(img_rgb)
    
    # 얼굴 마스크 생성 (눈, 코, 입 중심부만)
    FACE_KEYPOINTS = list(range(33, 173))  # 눈, 코, 입 주변만 추출
    
    overlap_ratio = 0
    face_detected = False
    
    if face_results.multi_face_landmarks:
        face_detected = True
        for lm in face_results.multi_face_landmarks:
            pts = [(int(lm.landmark[i].x * w), int(lm.landmark[i].y * h)) for i in FACE_KEYPOINTS if i < len(lm.landmark)]
            if pts:
                hull = cv2.convexHull(np.array(pts))
                cv2.fillConvexPoly(face_mask, hull, 255)
    
    # 손 마스크 생성 (padding 포함 확대)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            pts = np.array([(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark])
            if len(pts) > 0:
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(hand_mask, hull, 255)
        hand_mask = cv2.dilate(hand_mask, np.ones((25, 25), np.uint8))  # 손 마스크 확대
    
    # 겹침 계산
    face_area = cv2.countNonZero(face_mask)
    overlap_mask = cv2.bitwise_and(face_mask, hand_mask)
    overlap_area = cv2.countNonZero(overlap_mask)
    
    # 비율 계산
    if face_area > 0:
        overlap_ratio = overlap_area / face_area
        status = f"가림 비율: {overlap_ratio*100:.2f}%"
    else:
        status = "얼굴 미인식"
    
    # 시각화를 위한 마스크 컬러 이미지
    visualization = img_rgb.copy()
    
    # 얼굴 마스크 시각화 (반투명 오버레이)
    face_color_mask = np.zeros_like(img_rgb)
    face_color_mask[face_mask > 0] = [0, 255, 0]  # 녹색
    
    # 겹침 영역 시각화
    overlap_color_mask = np.zeros_like(img_rgb)
    overlap_color_mask[overlap_mask > 0] = [255, 0, 0]  # 빨간색
    
    # 손 영역 시각화
    hand_color_mask = np.zeros_like(img_rgb)
    hand_color_mask[hand_mask > 0] = [0, 0, 255]  # 파란색
    
    # 마스크 합치기 (겹침 영역이 가장 위에 표시되도록)
    alpha = 0.3
    visualization = cv2.addWeighted(visualization, 1, hand_color_mask, alpha, 0)
    visualization = cv2.addWeighted(visualization, 1, face_color_mask, alpha, 0)
    visualization = cv2.addWeighted(visualization, 1, overlap_color_mask, alpha, 0)
    
    # 결과 저장
    results.append({
        'filename': img_path,
        'face_detected': face_detected,
        'overlap_ratio': overlap_ratio,
        'visualization': visualization
    })
    
    # 그래프에 추가
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(visualization)
    plt.title(f"{img_path}\n{status}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("face_hand_overlap_results.png", dpi=150)
plt.show()

# 결과 출력
print("\n===== 분석 결과 =====")
for result in results:
    if result['face_detected']:
        print(f"{result['filename']}: 얼굴 중심부 중 {result['overlap_ratio']*100:.2f}% 가 손에 의해 가려졌습니다.")
    else:
        print(f"{result['filename']}: 얼굴을 인식하지 못했습니다.")
import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

# Initialize Mediapipe
mp_face = mp.solutions.face_mesh
mp_hand = mp.solutions.hands
face = mp_face.FaceMesh(static_image_mode=True)
hands = mp_hand.Hands(static_image_mode=True, max_num_hands=2)

# Find all PNG files in directory
png_files = glob.glob("*.png")
# Filter out the result image if it exists
png_files = [f for f in png_files if f != "face_hand_overlap_results.png"]

# List to store results
results = []

# Check if there are any PNG files
n_images = len(png_files)
if n_images == 0:
    print("No PNG files found in directory.")
    exit()

# Set up the figure for visualization
rows = int(np.ceil(n_images / 3))
cols = min(n_images, 3)

# Create a new figure
fig = plt.figure(figsize=(15, 5 * rows))

for idx, img_path in enumerate(png_files):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read {img_path}")
        continue
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Initialize masks
    face_mask = np.zeros((h, w), dtype=np.uint8)
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Process face and hand landmarks
    face_results = face.process(img_rgb)
    hand_results = hands.process(img_rgb)
    
    # Create face mask (using a better selection of face keypoints)
    # More complete set of facial keypoints - includes whole face area
    # Get all face keypoints to detect full face area
    overlap_ratio = 0
    face_detected = False
    
    if face_results.multi_face_landmarks:
        face_detected = True
        for lm in face_results.multi_face_landmarks:
            # Get all face mesh points for a more complete face detection
            # Focus on the central face area (keypoints for complete face oval)
            face_oval_points = list(range(0, 10)) + list(range(338, 400))
            # Add points for eyes, nose, and mouth area
            central_face_points = list(range(33, 173))
            # Combine all important face points
            all_face_points = face_oval_points + central_face_points
            
            pts = []
            for i in all_face_points:
                if i < len(lm.landmark):
                    x = int(lm.landmark[i].x * w)
                    y = int(lm.landmark[i].y * h)
                    pts.append((x, y))
            
            if pts:
                # Create convex hull of face points and fill it
                hull = cv2.convexHull(np.array(pts))
                cv2.fillConvexPoly(face_mask, hull, 255)
                
                # For visualization - highlight central face area
                central_pts = []
                for i in central_face_points:
                    if i < len(lm.landmark):
                        x = int(lm.landmark[i].x * w)
                        y = int(lm.landmark[i].y * h)
                        central_pts.append((x, y))
    
    # Create hand mask (with padding)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            pts = np.array([(int(p.x * w), int(p.y * h)) for p in hand_landmarks.landmark])
            if len(pts) > 0:
                hull = cv2.convexHull(pts)
                cv2.fillConvexPoly(hand_mask, hull, 255)
        hand_mask = cv2.dilate(hand_mask, np.ones((25, 25), np.uint8))  # Expand hand mask
    
    # Calculate overlap
    face_area = cv2.countNonZero(face_mask)
    overlap_mask = cv2.bitwise_and(face_mask, hand_mask)
    overlap_area = cv2.countNonZero(overlap_mask)
    
    # Calculate ratio
    if face_area > 0:
        overlap_ratio = overlap_area / face_area
        status = f"Overlap: {overlap_ratio*100:.2f}%"
    else:
        status = "No face detected"
    
    # Create visualization with colored masks
    visualization = img_rgb.copy()
    
    # Face mask visualization (translucent overlay)
    face_color_mask = np.zeros_like(img_rgb)
    face_color_mask[face_mask > 0] = [0, 255, 0]  # Green for face area
    
    # Overlap area visualization
    overlap_color_mask = np.zeros_like(img_rgb)
    overlap_color_mask[overlap_mask > 0] = [255, 0, 0]  # Red for overlap
    
    # Hand area visualization
    hand_color_mask = np.zeros_like(img_rgb)
    hand_color_mask[hand_mask > 0] = [0, 0, 255]  # Blue for hand area
    
    # Add explanation text about the colors
    explanation = f"Green: Face region | Blue: Hand region | Red: Overlap"
    
    # Combine masks (ensure overlap area is on top)
    alpha = 0.3
    visualization = cv2.addWeighted(visualization, 1, hand_color_mask, alpha, 0)
    visualization = cv2.addWeighted(visualization, 1, face_color_mask, alpha, 0)
    visualization = cv2.addWeighted(visualization, 1, overlap_color_mask, alpha, 0)
    
    # Store results
    results.append({
        'filename': img_path,
        'face_detected': face_detected,
        'overlap_ratio': overlap_ratio,
        'visualization': visualization
    })
    
    # Add to subplot
    plt.subplot(rows, cols, idx + 1)
    plt.imshow(visualization)
    plt.title(f"{img_path}\n{status}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("face_hand_overlap_results.png", dpi=150)
plt.close(fig)  # Close the figure to avoid displaying it twice

# Print results summary
print("\n===== Analysis Results =====")
for result in results:
    if result['face_detected']:
        print(f"{result['filename']}: {result['overlap_ratio']*100:.2f}% of face central area covered by hand.")
    else:
        print(f"{result['filename']}: No face detected.")
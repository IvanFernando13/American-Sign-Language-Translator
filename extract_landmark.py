import cv2
import mediapipe as mp
import os
import numpy as np
import json

TRAIN_DIR = "dataset/Train_Alphabet" 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def get_70_features(hand_landmarks, scale):
    wrist = hand_landmarks.landmark[0]
    features = []
    
    
    for lm in hand_landmarks.landmark:
        features.extend([(lm.x - wrist.x)/scale, (lm.y - wrist.y)/scale, (lm.z - wrist.z)/scale])
    
   
    tips = [4, 8, 12, 16, 20]
    for tip_idx in tips:
        tip = hand_landmarks.landmark[tip_idx]
        dist = np.sqrt((tip.x-wrist.x)**2 + (tip.y-wrist.y)**2 + (tip.z-wrist.z)**2)
        features.append(dist/scale)

    
    l8, l12, l4, l20 = hand_landmarks.landmark[8], hand_landmarks.landmark[12], hand_landmarks.landmark[4], hand_landmarks.landmark[20]
    dist_8_12 = np.sqrt((l8.x-l12.x)**2 + (l8.y-l12.y)**2 + (l8.z-l12.z)**2)
    dist_4_20 = np.sqrt((l4.x-l20.x)**2 + (l4.y-l20.y)**2 + (l4.z-l20.z)**2)
    
    features.extend([dist_8_12/scale, dist_4_20/scale])
    return features

def run_extraction():
    data, labels = [], []
    folders = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    label_map = {folder: i for i, folder in enumerate(folders)}
    
    for folder_name in folders:
        label_idx = label_map[folder_name]
        full_path = os.path.join(TRAIN_DIR, folder_name)
        images = [f for f in os.listdir(full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Extracting {folder_name}...")

        for img_name in images:
            img = cv2.imread(os.path.join(full_path, img_name))
            if img is None: continue
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                all_x = [l.x for l in lm.landmark]; all_y = [l.y for l in lm.landmark]
                scale = max(max(all_x)-min(all_x), max(all_y)-min(all_y)) or 1
                data.append(get_70_features(lm, scale))
                labels.append(label_idx)
                    
    np.save("X_train.npy", np.array(data))
    np.save("y_train.npy", np.array(labels))
    with open("label_map.json", "w") as f: json.dump(label_map, f)
    print("✅ 70-Feature Extraction Complete.")

if __name__ == "__main__":
    run_extraction()
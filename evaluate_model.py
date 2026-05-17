import numpy as np
import tensorflow as tf
import os
import cv2
import mediapipe as mp
import json
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. SETUP - Check your folder name in File Explorer!
# Based on your previous screenshots, try: "dataset/asl_alphabet_test"
TEST_DIR = "dataset/asl_alphabet_test" 

if not os.path.exists(TEST_DIR):
    print(f"❌ ERROR: Test directory '{TEST_DIR}' not found!")
    exit()

# Load Model and Labels
model = tf.keras.models.load_model("model/sign_model.h5")
with open("label_map.json", "r") as f:
    label_map = json.load(f)
inv_label_map = {i: v for v, i in label_map.items()}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def get_68_features(hand_landmarks, scale):
    wrist = hand_landmarks.landmark[0]
    features = []
    # 63 Coordinates
    for lm in hand_landmarks.landmark:
        features.extend([(lm.x - wrist.x)/scale, (lm.y - wrist.y)/scale, (lm.z - wrist.z)/scale])
    # 5 Key Distances
    for tip_idx in [4, 8, 12, 16, 20]:
        tip = hand_landmarks.landmark[tip_idx]
        dist = np.sqrt((tip.x-wrist.x)**2 + (tip.y-wrist.y)**2 + (tip.z-wrist.z)**2)
        features.append(dist/scale)
    return features

# 2. EVALUATION LOOP
y_true, y_pred = [], []
processed_count = 0
skipped_count = 0

print("🔍 Scanning test folders...")

for folder in sorted(os.listdir(TEST_DIR)):
    # Match the folder name to your training labels
    if folder not in label_map:
        print(f"⚠️ Skipping folder '{folder}' (Not in label_map.json)")
        continue
        
    label_idx = label_map[folder]
    path = os.path.join(TEST_DIR, folder)
    
    # Process each image in the subfolder
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            all_x = [l.x for l in lm.landmark]
            all_y = [l.y for l in lm.landmark]
            scale = max(max(all_x)-min(all_x), max(all_y)-min(all_y)) or 1
            
            feat = get_68_features(lm, scale)
            pred = model.predict(np.array([feat]), verbose=0)
            
            y_true.append(label_idx)
            y_pred.append(np.argmax(pred))
            processed_count += 1
        else:
            skipped_count += 1

print(f"\n✅ Done! Processed: {processed_count} images | Skipped: {skipped_count} (no hand found)")

# 3. GENERATE REPORTS
if len(y_true) > 0:
    print("\n--- CLASSIFICATION REPORT ---")
    # Only use labels that were actually found in the test set
    unique_labels = np.unique(y_true)
    target_names = [inv_label_map[i] for i in unique_labels]
    
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.title("ASL Research Evaluation Matrix")
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()
else:
    print("❌ Critical: No hands were detected in any images. Try images with clearer lighting.")
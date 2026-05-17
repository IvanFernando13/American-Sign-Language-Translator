import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import json

# 1. Load the 1000-epoch model and the dynamic label map
model = tf.keras.models.load_model("model/sign_model.h5")
with open("label_map.json", "r") as f:
    label_map = json.load(f)

# Reverse the map: {0: "A", 1: "B", 26: "space"}
index_to_label = {int(v): k for k, v in label_map.items()}

# 2. MediaPipe Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 3. Sentence Builder & Smoothing Variables
current_sentence = ""
last_added_label = ""
label_stability_counter = 0
STABILITY_THRESHOLD = 30  # Hold sign for ~1 second to "type" it
prediction_buffer = deque(maxlen=15) # Smooths out "flickering" predictions

cap = cv2.VideoCapture(0)

print("Interpreter Running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1) # Mirror for natural interaction
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    detected_label = "nothing" 

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the screen
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- SCALE INVARIANT EXTRACTION (Matches your training) ---
            landmark_list = []
            wrist = hand_landmarks.landmark[0]

            # Calculate hand scale
            all_x = [lm.x for lm in hand_landmarks.landmark]
            all_y = [lm.y for lm in hand_landmarks.landmark]
            scale = max(max(all_x) - min(all_x), max(all_y) - min(all_y))

            # Normalize landmarks
            for lm in hand_landmarks.landmark:
                landmark_list.extend([
                    (lm.x - wrist.x) / scale,
                    (lm.y - wrist.y) / scale,
                    (lm.z - wrist.z) / scale
                ])

            # Predict using the model
            prediction = model.predict(np.array([landmark_list]), verbose=0)
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            # Only count prediction if confidence is high enough
            if confidence > 0.7:
                detected_label = index_to_label.get(predicted_index, "nothing")

    # --- SENTENCE BUILDER LOGIC ---
    prediction_buffer.append(detected_label)
    most_common = Counter(prediction_buffer).most_common(1)[0][0]

    if most_common == last_added_label:
        label_stability_counter += 1
    else:
        last_added_label = most_common
        label_stability_counter = 0

    # Commit the letter to the sentence
    if label_stability_counter == STABILITY_THRESHOLD:
        if most_common == "space":
            current_sentence += " "
        elif most_common == "del":
            current_sentence = current_sentence[:-1]
        elif most_common != "nothing":
            current_sentence += most_common
        
        # Lock the counter so it doesn't repeat the letter immediately
        label_stability_counter = STABILITY_THRESHOLD + 1 

    # --- UI OVERLAY ---
    # Progress bar for stability
    progress = min(label_stability_counter / STABILITY_THRESHOLD, 1.0)
    cv2.rectangle(frame, (20, 440), (int(20 + (200 * progress)), 455), (0, 255, 0), -1)

    # Status Display
    cv2.putText(frame, f"Sign: {most_common}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Sentence Display
    cv2.rectangle(frame, (0, 460), (640, 480), (0, 0, 0), -1)
    cv2.putText(frame, f"Sentence: {current_sentence}", (10, 475), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("ASL Real-Time Interpreter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break #

cap.release()
cv2.destroyAllWindows()

# Inside the prediction block
prediction = model.predict(np.array([landmark_list]), verbose=0)
confidence = np.max(prediction)
predicted_index = np.argmax(prediction)

# ONLY update the label if the AI is more than 85% sure
if confidence > 0.85:
    detected_label = index_to_label.get(predicted_index, "nothing")
else:
    detected_label = "Thinking..." # Avoids flickering to a wrong letter
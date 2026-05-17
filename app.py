from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import json
import base64

app = Flask(__name__)

# Load core intelligence assets
model = tf.keras.models.load_model("model/sign_model.h5")
with open("label_map.json", "r") as f:
    label_map = json.load(f)
index_to_label = {int(v): k for k, v in label_map.items()}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

@app.route('/')
def index(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = frame.shape
    
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prediction_text = "None"
    hand_box = None 

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0]
        
        # Calculate dynamic UI bounding box coordinates
        x_c = [int(l.x * w) for l in lm.landmark]
        y_c = [int(l.y * h) for l in lm.landmark]
        hand_box = {'x': min(x_c)-20, 'y': min(y_c)-20, 'w': (max(x_c)-min(x_c))+40, 'h': (max(y_c)-min(y_c))+40}

        # Feature Generation Pipeline matching training shape (70 features)
        wrist = lm.landmark[0]
        scale = max(max([l.x for l in lm.landmark])-min([l.x for l in lm.landmark]), 
                    max([l.y for l in lm.landmark])-min([l.y for l in lm.landmark])) or 1
        
        f = []
        for l in lm.landmark: 
            f.extend([(l.x-wrist.x)/scale, (l.y-wrist.y)/scale, (l.z-wrist.z)/scale])
        for t in [4, 8, 12, 16, 20]:
            tip = lm.landmark[t]
            f.append(np.sqrt((tip.x-wrist.x)**2 + (tip.y-wrist.y)**2 + (tip.z-wrist.z)**2)/scale)
        
        d8_12 = np.sqrt((lm.landmark[8].x-lm.landmark[12].x)**2 + (lm.landmark[8].y-lm.landmark[12].y)**2)/scale
        d4_20 = np.sqrt((lm.landmark[4].x-lm.landmark[20].x)**2 + (lm.landmark[4].y-lm.landmark[20].y)**2)/scale
        f.extend([d8_12, d4_20])

        # Core Inference Check
        preds = model.predict(np.array([f]), verbose=0)[0]
        idx = np.argmax(preds)
        conf = preds[idx]
        label = index_to_label.get(idx)

        # Baseline: Trust the model if confidence is generally good
        if conf > 0.60:
            prediction_text = label

        # ====================================================================
        # THE FIX: HAND-AGNOSTIC GEOMETRIC GATEKEEPER
        # ====================================================================
        
        # 1. R vs U Override
        if label in ["U", "R"]:
            is_index_left_of_middle = lm.landmark[5].x < lm.landmark[9].x
            if is_index_left_of_middle:
                prediction_text = "R" if lm.landmark[8].x > lm.landmark[12].x else "U"
            else:
                prediction_text = "R" if lm.landmark[8].x < lm.landmark[12].x else "U"
                
        # 2. W and Y Consistency Check
        elif label == "V" and f[66] > 0.55: 
            prediction_text = "W"
        elif label in ["I", "Y", "None"] and f[69] > 0.8: 
            prediction_text = "Y"

        # 3. Robust X, T, and D Separator (Kinematic Straightness Ratio)
        elif label in ["X", "T", "D", "S"] or conf < 0.85:
            
            # Shield 1: Ensure the back three fingers are curled into a fist
            middle_curled = lm.landmark[12].y > lm.landmark[9].y
            ring_curled = lm.landmark[16].y > lm.landmark[13].y
            pinky_curled = lm.landmark[20].y > lm.landmark[17].y
            
            # Ensure index finger is raised (tip above knuckle)
            index_raised = lm.landmark[8].y < lm.landmark[5].y

            if middle_curled and ring_curled and pinky_curled and index_raised:
                
                # Shield 2: T-Shield (Gap between Thumb Tip and Index PIP Joint)
                thumb_index_gap = np.sqrt((lm.landmark[6].x - lm.landmark[4].x)**2 + (lm.landmark[6].y - lm.landmark[4].y)**2) / scale
                
                # Shield 3: X/D Separator (Kinematic Straightness Ratio)
                # Path length: Sum of the 3 individual finger segments
                d5_6 = np.sqrt((lm.landmark[5].x - lm.landmark[6].x)**2 + (lm.landmark[5].y - lm.landmark[6].y)**2)
                d6_7 = np.sqrt((lm.landmark[6].x - lm.landmark[7].x)**2 + (lm.landmark[6].y - lm.landmark[7].y)**2)
                d7_8 = np.sqrt((lm.landmark[7].x - lm.landmark[8].x)**2 + (lm.landmark[7].y - lm.landmark[8].y)**2)
                path_length = d5_6 + d6_7 + d7_8
                
                # Vector length: Direct line from Knuckle (5) to Tip (8)
                vec_length = np.sqrt((lm.landmark[5].x - lm.landmark[8].x)**2 + (lm.landmark[5].y - lm.landmark[8].y)**2)
                
                straightness_ratio = vec_length / path_length if path_length > 0 else 0

                # Routing Logic
                if thumb_index_gap <= 0.15:
                    prediction_text = "T"
                elif straightness_ratio > 0.88:
                    prediction_text = "D"
                else:
                    prediction_text = "X"
                    
        # ====================================================================

    return jsonify({'prediction': prediction_text, 'box': hand_box})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
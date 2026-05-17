import cv2
import os

# Configuration
DATA_DIR = "dataset/Train_Alphabet" 
labels = ['R', 'U', 'V', 'W', 'Y'] # The "Final Boss" Cluster

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

cap = cv2.VideoCapture(0)

for label in labels:
    print(f"\n--- Collecting: {label} ---")
    label_path = os.path.join(DATA_DIR, label)
    os.makedirs(label_path, exist_ok=True)

    count = len(os.listdir(label_path))
    limit = count + 400 

    while count < limit:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) # Mirror mode

        display = frame.copy()
        cv2.putText(display, f"Letter: {label} | {count}/{limit}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv2.imshow("Data Collection", display)

        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(os.path.join(label_path, f"final_{count}.jpg"), frame)
            count += 1
        elif key == ord('q'): break

cap.release()
cv2.destroyAllWindows()
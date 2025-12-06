import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# -------------------- SETTINGS --------------------
MODEL_PATH = "best_gesture_model.h5"  # Your trained model
NUM_CLASSES = 6
# ---------------------------------------------------

# Load your trained model
model = load_model(MODEL_PATH)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    prediction_text = "No hand detected"

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw landmarks on frame
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract landmarks as input for model
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks).reshape(1, -1)

        # Predict gesture
        pred = model.predict(landmarks)
        gesture = np.argmax(pred)
        confidence = np.max(pred)

        prediction_text = f"Gesture: {gesture} ({confidence*100:.1f}%)"

    # Display prediction
    cv2.putText(frame, prediction_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

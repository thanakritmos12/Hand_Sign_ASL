import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

import warnings
# Suppress all UserWarnings (use with caution)
warnings.simplefilter("ignore", UserWarning)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

characters = [chr(i) for i in range(65, 91)]  # A-Z
char_index = 0
hold_start_time = None

def detect_hand_sign(frame):
    global char_index, hold_start_time
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []
    predicted_character = None

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 1:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                           mp_drawing_styles.get_default_hand_landmarks_style(),
                                           mp_drawing_styles.get_default_hand_connections_style())

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize hand landmark positions
                min_x, min_y = min(x_), min(y_)
                data_aux = [(x - min_x) for x in x_] + [(y - min_y) for y in y_]

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = chr(int(prediction[0]) + 65)

                # Show predicted character on frame
                cv2.putText(frame, f'Detected: {predicted_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Please show only one hand!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

    # Show current character to perform
    cv2.putText(frame, f'Do "{characters[char_index]}"', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if predicted_character == characters[char_index]:
        if hold_start_time is None:
            hold_start_time = time.time()

        elapsed_time = time.time() - hold_start_time

        cv2.putText(frame, f'Hold for 2 seconds. Time: {int(elapsed_time)}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if elapsed_time >= 2:
            # Display 'WELL DONE' on the frame
            cv2.putText(frame, 'WELL DONE!', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6, cv2.LINE_AA)

            # Move to the next character
            char_index += 1
            if char_index >= len(characters):
                char_index = 0  # Reset after reaching the last character

            hold_start_time = None
    else:
        hold_start_time = None

    return frame

def run_alphabet_lesson():
    cap = cv2.VideoCapture(0)
    
    # Set resolution to reduce processing load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0  # To control the frame rate

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add a small delay to control frame rate
        time.sleep(0.01)  # Adjust delay as needed

        frame = detect_hand_sign(frame)

        # Reduce the frequency of frames sent to the client
        if frame_count % 5 == 0:  # Send every 5th frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1

    cap.release()

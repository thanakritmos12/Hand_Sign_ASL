import warnings

# Suppress all UserWarnings (use with caution)
warnings.simplefilter("ignore", UserWarning)

import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the sequence of characters to display
characters = [chr(i) for i in range(65, 91)]  # A-Z

# Initialize index for characters
char_index = 0
success_count = 0
hold_start_time = None  # Track when to start counting for hold time

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Prepare the input data for prediction
    data_aux = []
    x_ = []
    y_ = []

    # Default value for predicted_character
    predicted_character = None

    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 1:  # Only process if one hand is detected
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on hand
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,  # Draw connections between landmarks
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_index = int(prediction[0])  # Convert to integer
            predicted_character = chr(predicted_index + 65)  # Convert to corresponding letter (A-Z)

            # Show predicted character on frame
            cv2.putText(frame, f'Detected: {predicted_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # If more than one hand is detected, display a warning
            cv2.putText(frame, 'Please show only one hand!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show current character to perform
    cv2.putText(frame, f'Do "{characters[char_index]}"', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Check if the predicted character matches the current character
    if predicted_character == characters[char_index]:
        if hold_start_time is None:  # Start timing
            hold_start_time = time.time()
        
        elapsed_time = time.time() - hold_start_time

        # Show the elapsed time on the frame
        cv2.putText(frame, f'Hold for 2 seconds. Time: {int(elapsed_time)}s', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if elapsed_time >= 2:  
            # Show "WELL DONE" message after successful detection
            cv2.putText(frame, 'WELL DONE!', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            cv2.waitKey(2000)  # Show message for 2 seconds

            # Move to the next character
            char_index += 1
            if char_index >= len(characters):
                break  # Exit if all characters have been processed

            # Reset the hold time
            hold_start_time = None  # Reset for the next character

    else:
        # Reset the hold time if the character does not match
        hold_start_time = None  # Reset for the next character

    # Show the frame
    cv2.imshow('frame', frame)

    # Check for key press to skip character (n) or exit (q)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        char_index += 1
        if char_index >= len(characters):
            break  # Exit if all characters have been processed

cap.release()
cv2.destroyAllWindows()

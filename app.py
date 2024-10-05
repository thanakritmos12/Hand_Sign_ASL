from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import pickle
import mediapipe as mp
import time
import warnings

# Suppress all UserWarnings (use with caution)
warnings.simplefilter("ignore", UserWarning)

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {i: chr(i + 65) for i in range(26)}  # A-Z mapping

# Store the final predicted results
last_prediction = ''

def generate_frames():
    global last_prediction  # ใช้ global เพื่อเก็บค่า prediction
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
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
            predicted_character = labels_dict[int(prediction[0])]

            # อัพเดตตัวแปร last_prediction
            last_prediction = predicted_character

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def generate_frames_alt():
    global last_predict_alt, last_instruction, hold_start_time, char_index  # ใช้ global เพื่อเก็บค่า prediction และ instruction
    cap = cv2.VideoCapture(0)
    characters = [chr(i) for i in range(65, 91)]  # A-Z
    char_index = 0  # เริ่มจากตัว A

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_ = []
        y_ = []
        predicted_character = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
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
            predicted_character = labels_dict[int(prediction[0])]

            # Update the last_predict_alt
            last_predict_alt = predicted_character

        # Logic to check if the detected character matches the current target
        target_character = characters[char_index]  # Target character
        if predicted_character == target_character:
            if hold_start_time is None:
                hold_start_time = time.time()  # Start timing when the character is shown

            elapsed_time = time.time() - hold_start_time  # Calculate time held

            if elapsed_time >= 2:  # If character is held for 2 seconds
                # Update instruction to move to the next character
                if char_index + 1 < len(characters):
                    last_instruction = f'Do "{characters[char_index + 1]}"'  # Move to next character
                    char_index += 1
                else:
                    last_instruction = "Finished!"  # No more characters left to show
                hold_start_time = None  # Reset the timer
        else:
            hold_start_time = None  # Reset the timer if the character doesn't match

        # Show the current instruction
        last_instruction = f'Do "{target_character}"' if hold_start_time is None else last_instruction

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/lesson1')
# def lesson1():
#     return render_template('lesson1.html')

@app.route('/lesson1')
def lesson1():
    return render_template('lesson1.html', lesson_number=1)


@app.route('/lesson2')
def lesson2():
    return render_template('lesson2.html', lesson_number=2)

@app.route('/video_feed/<int:lesson_number>')
def video_feed(lesson_number):
    if lesson_number == 1:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    elif lesson_number == 2:
        return Response(generate_frames_alt(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid lesson number", 404

@app.route('/get_asl_result')
def get_asl_result():
    global last_prediction
    return {'result': last_prediction}

@app.route('/get_asl_result_alt')
def get_asl_result_alt():
    global last_predict_alt
    return {'result': last_predict_alt}

@app.route('/get_asl_instruction', methods=['GET'])
def get_asl_instruction():
    global last_instruction  # ใช้ global เพื่อดึงคำสั่งล่าสุด
    return jsonify({'instruction': last_instruction})


if __name__ == '__main__':
    app.run(debug=True)

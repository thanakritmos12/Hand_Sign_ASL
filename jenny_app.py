import streamlit as st
import cv2
import numpy as np
import pickle
import time
import mediapipe as mp
import os
import pandas as pd  # Import pandas to work with CSV

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the sequence of characters to display
characters = [chr(i) for i in range(65, 91)]  # A-Z
char_index = 0
hold_start_time = None

labels_dict = {i: chr(i + 65) for i in range(26)}

# Data storage for logging
log_data = []  # Create a list to store log data


def main():
    if not st.session_state.get('logged_in', False):  # Show login page if not logged in
        login_page()
    elif st.session_state.get('mode') == 'practice_mode':  # Show practice mode if selected
        practice_mode()
    elif st.session_state.get('mode') == 'character_detection':  # Show detection mode if selected
        character_detection_mode()
    else:  # Show main menu if logged in and no mode selected
        main_menu()


# Login page
def login_page():
    st.title("Log in")
    username = st.text_input("Enter your username:")
    if st.button("Login"):
        if username:  # Check if the user has entered a username
            st.session_state["username"] = username
            st.session_state["logged_in"] = True
        else:
            st.error("Please enter a username.")


def main_menu():
    st.title("Main Menu")
    st.subheader("Please choose a mode to begin your learning:")
    
    # Button to go to Practice Mode
    if st.button("Practice Mode"):
        st.session_state['mode'] = 'practice_mode'  # Set mode to practice

    # Button to go to Character Detection Mode
    if st.button("Character Detection Mode"):
        st.session_state['mode'] = 'character_detection'  # Set mode to detection

    # Button to log out
    if st.button("Log Out"):
        logout()


def practice_mode():
    st.title("Practice Mode")

    # Button to go back to the main menu
    if st.button("Back to Menu"):
        st.session_state['mode'] = None  # Clear the mode

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open camera")
        return

    frame_placeholder = st.empty()  # Placeholder for displaying the video frame
    image_placeholder = st.empty()  # Placeholder for displaying the character image
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

    # Status of the practice mode
    if 'running' not in st.session_state:
        st.session_state.running = False

    if st.button('Start Practice'):
        st.session_state.running = True
        # Display the character image immediately
        image_path = "images/mix.png"  # Replace with your actual image path
        if os.path.exists(image_path):
            image_placeholder.image(image_path, caption='Character', use_column_width=True)
        else:
            image_placeholder.write("Character image not found.")

    if st.button('Stop Practice'):
        st.session_state.running = False
        image_placeholder.empty()  # Clear the image when practice is stopped

    while st.session_state.running:
        ret, frame = cap.read()  # Read frame from camera
        if not ret:
            st.error("Failed to capture video")
            break

        # Use the hand detection function
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Prepare input for prediction
                data_aux = []
                x_ = [landmark.x for landmark in hand_landmarks.landmark]
                y_ = [landmark.y for landmark in hand_landmarks.landmark]

                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(x_[i] - min(x_))
                    data_aux.append(y_[i] - min(y_))

                # Predict the character
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = chr(int(prediction[0]) + 65)  # Convert index to character

                # Draw bounding box and show prediction
                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10
                x2 = int(max(x_) * frame.shape[1]) - 10
                y2 = int(max(y_) * frame.shape[0]) - 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Display the video frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")  # Show the frame in Streamlit

    cap.release()
    cv2.destroyAllWindows()


def character_detection_mode():
    st.title("Character Detection Mode")

    # Button to go back to the main menu
    if st.button("Back to Menu"):
        st.session_state['mode'] = None  # Clear the mode

    global hold_start_time

    # Initialize session_state variables if not exist
    if 'run' not in st.session_state:
        st.session_state['run'] = False
    if 'char_index' not in st.session_state:
        st.session_state['char_index'] = 0  # Start at first character (A)
    if 'score' not in st.session_state:
        st.session_state['score'] = 0  # Initialize score
    if 'log_data' not in st.session_state:
        st.session_state['log_data'] = []  # Initialize log data

    # Start video capture
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

    # Create a static layout using columns
    col1, col2 = st.columns([1, 2])  # Adjust columns: col1 for images, col2 for video frame

    with col1:
        image_placeholder = st.empty()  # Placeholder for displaying the character image

    with col2:
        frame_placeholder = st.empty()  # For the video frame
        frame_width = 640  # Set a fixed width for the video frame
        frame_height = 480  # Set a fixed height for the video frame

        well_done_container = st.container()
        well_done_placeholder = well_done_container.empty()  # Placeholder for WELL DONE message

        instruction_placeholder = st.empty()   # For the instruction: Do "A", "B", etc.
        detection_placeholder = st.empty()     # For the detected character
        timer_placeholder = st.empty()         # For the timer feedback
        score_placeholder = st.empty()         # For displaying the score
        final_score_placeholder = st.empty()   # For displaying the final score when stopping

    # Control the video stream with a button
    if st.button('Start Detection'):
        st.session_state.run = True
        st.session_state.char_index = 0  # Reset to A when starting detection
        st.session_state.score = 0  # Reset score
        hold_start_time = None  # Reset hold start time

        # Show the initial character image (A)
        image_path = f"images/{characters[st.session_state['char_index']]}.png"
        if os.path.exists(image_path):
            image_placeholder.image(image_path, caption=f'Character: {characters[st.session_state["char_index"]]}', use_column_width=True)

    elif st.button('Stop Detection'):
        st.session_state.run = False
        final_score_placeholder.write(f'**Final Score: {st.session_state["score"]:.2f}%**')  # Display final score
        
        # Log the results
        log_entry = {
            'Username': st.session_state['username'],
            'Mode': st.session_state['mode'],
            'Score': st.session_state['score']
        }
        st.session_state['log_data'].append(log_entry)  # Add the entry to the log data

        # Save log data to CSV
        log_df = pd.DataFrame(st.session_state['log_data'])
        log_df.to_csv('user_log.csv', mode='a', index=False, header=not os.path.exists('user_log.csv'))  # Save to CSV file

    # Start the detection loop if the detection has started
    if st.session_state.run:
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            predicted_character = None  # Initialize the predicted character

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    data_aux = []
                    x_ = [landmark.x for landmark in hand_landmarks.landmark]
                    y_ = [landmark.y for landmark in hand_landmarks.landmark]

                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(x_[i] - min(x_))
                        data_aux.append(y_[i] - min(y_))

                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = chr(int(prediction[0]) + 65)  # Convert index to character

                    # Draw bounding box and show prediction
                    x1 = int(min(x_) * frame.shape[1]) - 10
                    y1 = int(min(y_) * frame.shape[0]) - 10
                    x2 = int(max(x_) * frame.shape[1]) - 10
                    y2 = int(max(y_) * frame.shape[0]) - 10

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                    # Check if the predicted character matches the current character
                    if predicted_character == characters[st.session_state.char_index]:
                        if hold_start_time is None:
                            hold_start_time = time.time()  # Start timing
                        elapsed_time = time.time() - hold_start_time

                        # Display elapsed time on the screen
                        timer_placeholder.write(f'**Hold for 2 seconds. Elapsed time: {elapsed_time:.1f}s**')

                        # Update score and show WELL DONE if held for 2 seconds
                        if elapsed_time >= 2:
                            well_done_placeholder.markdown(f'<h1 style="color: green; text-align: center;">WELL DONE!</h1>', unsafe_allow_html=True)
                            st.session_state.score += 3.85  # Increment score
                            time.sleep(2)  # Pause for a moment

                            # Move to the next character
                            st.session_state.char_index += 1
                            if st.session_state.char_index >= len(characters):
                                st.session_state.char_index = 0  # Loop back to A

                            # Update the image for the new character
                            image_path = f"images/{characters[st.session_state['char_index']]}.png"
                            if os.path.exists(image_path):
                                image_placeholder.image(image_path, caption=f'Character: {characters[st.session_state["char_index"]]}', use_column_width=True)

                            # Update score display
                            score_placeholder.write(f'**Current Score: {st.session_state["score"]:.2f}%**')  # Display current score

                            # Reset hold time and clear WELL DONE message
                            hold_start_time = None
                            well_done_placeholder.empty()
                    else:
                        hold_start_time = None
                        timer_placeholder.empty()  # Clear the timer if no correct detection

            # Resize and display the video frame with fixed dimensions
            resized_frame = cv2.resize(frame, (frame_width, frame_height))
            frame_placeholder.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

            # Display current instruction (Do: A, B, C,...)
            instruction_placeholder.write(f'### Do: "{characters[st.session_state["char_index"]]}"')

            # Display detected character (if any)
            if predicted_character:
                detection_placeholder.write(f'**Detected: {predicted_character}**')
            else:
                detection_placeholder.write('Detected: None')

    cap.release()
    cv2.destroyAllWindows()


def logout():
    # Clear session state and redirect to login page
    st.session_state.clear()  # Clear all session data
    st.session_state['logged_in'] = False  # Reset logged in status
    main()  # Redirect to the main page (login page)


if __name__ == "__main__":
    main()

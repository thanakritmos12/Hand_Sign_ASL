import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26
dataset_size = 300

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Get the list of existing files in the directory
    existing_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
    if existing_files:
        # Extract the numbers from the filenames and find the maximum
        existing_numbers = [int(f.split('.')[0]) for f in existing_files]
        next_image_number = max(existing_numbers) + 1
    else:
        next_image_number = 1  # Start from 1 if no images are present

    # Display ready message
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        
        # Save the image with the correct numbering
        file_name = f'{next_image_number}.jpg'
        cv2.imwrite(os.path.join(class_dir, file_name), frame)

        next_image_number += 1
        counter += 1

cap.release()
cv2.destroyAllWindows()














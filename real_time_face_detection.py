import face_recognition
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json
import tensorflow
print(cv2.__version__)
print(face_recognition.__version__)
print(tensorflow.__version__)
print(keras.__version__)


# Get the webcam #0 (the default one, 1, 2 etc means additional attached cams)
webcam_video_stream = cv2.VideoCapture(0)

# Initialize the array variable to hold all face locations in the frame
all_face_locations = []

while True:
    # Get the current frame from video stream as an image
    ret, current_frame = webcam_video_stream.read()
    
    # Check if the frame was successfully read
    if not ret:
        print("Can not take the frame from the webcam. please check again")
        break

    # Resize the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    
    # Detect all faces in the image
    all_face_locations = face_recognition.face_locations(current_frame_small,number_of_times_to_upsample = 2,model='hog')
    
    # Looping through the face locations
    for index, current_face_location in enumerate(all_face_locations):
        # Split the tuple (4 positions for every face)
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top_pos *= 4
        right_pos *= 4
        bottom_pos *= 4
        left_pos *= 4
        
        # Print the location of the current face
        print('Found face {} at the locations Top: {}, Left: {}, Bottom: {}, Right: {}'.format(index+1, top_pos, right_pos, bottom_pos, left_pos))
        
        current_face_image = current_frame[top_pos:bottom_pos, left_pos: right_pos]
        current_face_image = cv2.GaussianBlur(current_face_image, (99, 99), 30)

        current_frame[top_pos: bottom_pos, left_pos: right_pos] = current_face_image 
        # Draw rectangle around each face location in the main video frame inside the loop
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)

    # Showing the current frame with rectangle drawn
    cv2.imshow("Webcam video", current_frame)
    
    # Press 'q' on the keyboard to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the stream and cam
webcam_video_stream.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
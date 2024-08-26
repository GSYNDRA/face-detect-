#importing the required libraries
import cv2
import face_recognition
import dlib

#capture the video from default camera 
webcam_video_stream = cv2.VideoCapture(0)

#load the pretrained HOG SVN model
face_detection_classifier = dlib.get_frontal_face_detector()

# load shape predictor to predict face landmark points of individual facial structures
face_shape_predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat')

#load the sample images and get the 128 face embeddings from them
trump_image = face_recognition.load_image_file('images/linh.jpg')
trump_face_encodings = face_recognition.face_encodings(trump_image)[0]

modi_image = face_recognition.load_image_file('images/minh.png')
modi_face_encodings = face_recognition.face_encodings(modi_image)[0]

abhi_image = face_recognition.load_image_file('images/giang.jpg')
abhi_face_encodings = face_recognition.face_encodings(abhi_image)[0]

#save the encodings and the corresponding labels in separate arrays in the same order
known_face_encodings = [modi_face_encodings, trump_face_encodings, abhi_face_encodings]
known_face_names = ["Minh", "Linh", "Giang"]

#initialize the array variable to hold all face locations, encodings and names 
all_face_locations = []
all_face_encodings = []
all_face_names = []

#loop through every frame in the video
while True:
    #get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()

    #detect all face locations using the HOG SVN classifier
    all_face_locations = face_detection_classifier(current_frame, 1)    

    #initialize list for aligned faces
    aligned_faces = []

    #check if any faces are detected
    if len(all_face_locations) > 0:
        # loop through each face detected
        for current_face_location in all_face_locations:
            # predict landmarks for each face
            landmarks = face_shape_predictor(current_frame, current_face_location)
            # align face using the detected landmarks
            aligned_face = dlib.get_face_chip(current_frame, landmarks, size=160, padding=0.25)
            aligned_faces.append(aligned_face)

    # Detect face encodings for all aligned faces
    all_face_encodings = []
    for face in aligned_faces:
        face_encodings = face_recognition.face_encodings(face)
        if len(face_encodings) > 0:
            all_face_encodings.append(face_encodings[0])

    # Looping through the face locations and the face embeddings
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        # Splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = (current_face_location.top(), current_face_location.right(), 
                                                     current_face_location.bottom(), current_face_location.left())

        # Find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding, tolerance=0.4)
        
        # String to hold the label
        name_of_person = 'Unknown face'
        
        # Check if the all_matches have at least one item
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]
        
        # Draw rectangle around the face    
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)
        
        # Display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

    # Display the full frame with detected faces
    cv2.imshow("Webcam Video", current_frame)
    
    # Display each aligned face in a separate window
    for index, aligned_face in enumerate(aligned_faces):
        cv2.imshow(f"Aligned Face {index + 1}", aligned_face)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#release the stream and cam
#close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()

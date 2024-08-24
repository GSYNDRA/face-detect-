import cv2
import face_recognition 

# load the image to detect 
image_test = cv2.imread("images/trump-modi.jpg")

print(cv2. __version__)
print(face_recognition. __version__)

#   display the image to detect 
cv2.imshow("test", image_test)
# wait the locked image
# cv2.waitKey(0)

# Find and print total number of faces
# find all face locations using face_locations() function
# model can be 'cnn' or 'hog'

# detect all faces in the image
all_face_locations = face_recognition.face_locations(image_test, model='hog')
print('there are {} no of faces in this image'.format(len(all_face_locations)))

# looping through the face locations 
for index, current_face_location in enumerate(all_face_locations): 
    #split the tuple (4 positions for every face)
    top_pos, right_pos, bottom_pos, left_pos = current_face_location
    print('Found face {} at the locations Top: {}, Left: {}, Bottom: {}, Right: {}'.format(index+1, top_pos, right_pos, bottom_pos, left_pos))
    
    #Slice frame image array by positions
    current_face_image = image_test[top_pos:bottom_pos, left_pos:right_pos]

    #Blur the current face image
    current_face_image = cv2.GaussianBlur(current_face_image,(99, 99), 30)

    #Put the blurred face region back into the frame image
    image_test[top_pos:bottom_pos, left_pos:right_pos] = current_face_image
    # cv2.imshow('face no: '+str(index+1), current_face_image)
cv2.imshow('test', image_test)
cv2.waitKey(0)

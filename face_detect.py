import cv2 as cv

img=cv.imread('C:/Users/pratyush mishra/Desktop/py tutorial/projects(openCV)/jatin.jpeg')
# vid=cv.VideoCapture('C:/Users/pratyush mishra/Desktop/py tutorial/open cv/Resources/Videos/dog.mp4')


# # Replace 'output_video.mp4' with the desired name for the output video
# output_file = 'output_video.mp4'

# # Define the codec to use (H.264 is a common choice)
# fourcc = cv.VideoWriter_fourcc(*'mp4v')

# # Get the frames per second and frame size from the input video
# fps = int(vid.get(5))
# frame_width = int(vid.get(3))
# frame_height = int(vid.get(4))

# # Create the VideoWriter object
# out = cv.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height), isColor=False)

haar_cascade=cv.CascadeClassifier('C:/Users/pratyush mishra/Desktop/py tutorial/projects(openCV)/haar_face.xml')

# while True:
#     ret,frame=vid.read()
#     if not ret:
#         break

#     # Convert the frame to grayscale
#     gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#     # Perform face detection on the grayscale frame
#     faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

#     # Draw rectangles around the detected faces
#     for (x, y, w, h) in faces:
#         cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


#     # Write the grayscale frame to the output video
#     out.write(gray_frame)

    # # Display the grayscale frame (optional)
    # cv.imshow('Grayscale Video', gray_frame)
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break


gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)


faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=10)

#haar_cascade: This is presumably an instance of a Haar cascade classifier. A Haar cascade is a machine learning object detection method used to identify objects or features in images.
# In this case, it's specifically designed for detecting faces.

# detectMultiScale: This is a method or function provided by the Haar cascade classifier. 
# It's used to detect objects (in this case, faces) within the provided image.

print(f'Number of faces found={len(faces_rect)}')

for(x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w, y+h),color=(0,255,0),thickness=2)

cv.imshow('detected',img)

cv.waitKey(0)
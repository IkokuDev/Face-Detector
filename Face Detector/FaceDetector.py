import cv2

#Load preTrained data from OpenCV
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
trained_smile_data = cv2.CascadeClassifier('haarcascade_smile.xml')


#Choose an image to detect faces in
#img = cv2.imread('Faces.jpg')

#To capture video from webcam(to use a video, replace 0 with "video_path.mp4")
webcam = cv2.VideoCapture(0)

#Iterate over frames
while True:
    successful_frame_read, frame = webcam.read()

    #make image black and white
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    

    

    #Draw rectangles around the faces
    for(x,y,w,h) in face_coordinates:
      cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

      #get the sub frame using numpy n-dimensional slicing
      the_face = frame[y:y+h , x:x+w]
      
      face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

      smiles = trained_smile_data.detectMultiScale(face_grayscale, 1.7, 20)

      #Label the face as smiling
      if len(smiles) > 0:
        cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

      #find smile in face
     # for(x_, y_, w_, h_) in smiles:
        # draw rectangles around smile
        #cv2.rectangle(the_face, (x_,y_), (x_+w_, y_+h_), (50, 50, 200), 3)

   
    cv2.imshow('Face Detector by M!ChV3L', frame)
    key = cv2.waitKey(1)

    if key ==81 or key==113:
        break
#release the videoCapture object
webcam.release()
cv2.destroyAllWindows()

print("Code Ran Successfully")
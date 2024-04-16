import cv2
from deepface import DeepFace

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("test43.mp4")
width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH ))
height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT ))
fps =  input_movie.get(cv2.CAP_PROP_FPS)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# Load some sample pictures and learn how to recognize them.

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

while True:
    # Capture frame-by-frame
    ret, frame = input_movie.read()
    frame_number += 1

    if not ret:
        break

    # Detect faces in the frame
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0,255,255)]
    ems = ["angry", "fear", "neutral", "disgust", "sad","happy","surprised"]

    # For each detected face, predict emotion
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        col = (0,0,0)
        if isinstance(result, list) and len(result) > 0:
            emotion = max(result[0]['emotion'].items(), key=lambda x: x[1])[0]
            for em in len(ems):
                if emotion == ems[em]:
                    col = cols[em]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
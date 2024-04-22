import cv2
from deepface import DeepFace
import os
from PIL import Image
import pandas as pd

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
diff_count = 0

cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 0), (0,255,255)]
peeps = []

while True:
    # Capture frame-by-frame
    ret, frame = input_movie.read()
    frame_number += 1

    if not ret:
        break

    # Detect faces in the frame
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    output_folder = "/tmp/temp_dir"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.chmod(output_folder, 0o777)

    # For each detected face, predict emotion
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        bgr_frame = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

        disty = 10000
        id = 0
        k = 1000
        while id == 0:
            disty = 10000
            id = 0
            items = os.listdir(output_folder)
            # Check if the length of the list is zero (i.e., directory is empty)
            if len(items) != 0:
                fi = DeepFace.find(bgr_frame, output_folder,"VGG-Face","cosine",0,"opencv",1,0,0.6,"base",0)
                for df in fi:
                    identities = df['identity'].to_list()
                    disties = df['distance'].to_list()
                    for index, value in enumerate(disties):
                        if value < disty:
                            id = identities[index]
                            if id not in peeps:
                                peeps.append(id)
                            k = peeps.index(id)
            if id == 0:
                # Save the frame as an image in the output folder
                output_path = os.path.join(output_folder, f"frame_{frame_number}_{(x, y, w, h)}.png")
                image = Image.fromarray(bgr_frame)
                image.save(output_path)
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        col = cols[k]
        if isinstance(result, list) and len(result) > 0:
            emotion = max(result[0]['emotion'].items(), key=lambda x: x[1])[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), col, 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

# All done!
input_movie.release()
cv2.destroyAllWindows()
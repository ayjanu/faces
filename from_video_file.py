import face_recognition
import cv2
from math import sqrt

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("hamilton_clip.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))

# Load some sample pictures and learn how to recognize them.
lmm_image = face_recognition.load_image_file("lin-manuel-miranda.png")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file("alex-lacamoire.png")
al_face_encoding = face_recognition.face_encodings(al_image)[0]

known_faces = [
    lmm_face_encoding,
    al_face_encoding
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0


def isCloseEnough(old_point, new_point):
    x1,y1 = old_point
    x2,y2 = new_point

    dist = sqrt((x1-x2)**2 + (y1-y2)**2)
    

def getMidPoint(topLeft, bottomRight):
    x1,y1 = topLeft
    x2,y2 = bottomRight
    return (x1+x2)/2, (y1+y2)/2

previous_vector = []
percent_threshold = 0.05

while True:
    curr_vector = []

    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    code = cv2.COLOR_BGR2RGB
    rgb_frame = cv2.cvtColor(rgb_frame, code)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "Lin-Manuel Miranda"
        elif match[1]:
            name = "Alex Lacamoire"

        face_names.append(name)

    # Label the results
    for i, ((top, right, bottom, left), name) in enumerate(zip(face_locations, face_names)):
        print(top)
        
        midPoint = getMidPoint((left, top), (right, bottom))
        curr_vector.append((midPoint, name))

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        # box but no name
        if not name:
            print("box but no name", frame_number )
            cv2.imwrite(f'pics_no_name/frame_{frame_number}.jpg', frame)
        if previous_vector:
            for item in previous_vector:
                if len(item) == 2:
                    old_point, old_name = item
                    for new_point, new_name in curr_vector:
                        if isCloseEnough(old_point, midPoint): 
                            name = old_name
                            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            continue
        # box and name
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # no box, no name
    if len(face_locations) == 0:
        print("no box and no name")
        cv2.imwrite(f'pics_no_box_no_name/frame_{frame_number}.jpg', frame)

    previous_vector = curr_vector


    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

    previous_vector = face_locations


# All done!
input_movie.release()

cv2.destroyAllWindows()







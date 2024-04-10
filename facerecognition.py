import cv2
from deepface import DeepFace

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to capture frame")
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
            cv2.rectangle(frame, (x, y), (x+w, y+h), col, 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            print("Face not detected or error in analysis:", result)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
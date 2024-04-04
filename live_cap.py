import cv2
import numpy as np
import pyautogui
import face_recognition

# Get screen resolution
screen_width, screen_height = pyautogui.size()

# Load a sample image and learn how to recognize it.
known_image = face_recognition.load_image_file("known_faces/Jatin.png")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Set the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (screen_width, screen_height))

# Capture screen and display the screen recording in real-time
while True:
    # Capture screen image
    screen = pyautogui.screenshot()
    frame = np.array(screen)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    # Draw rectangles around the faces and label them if recognized
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches the known face
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        name = "Unknown"
        if matches[0]:
            name = "Known Person"
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Label the face
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Write the frame
    out.write(frame)
    
    # Display the frame
    cv2.imshow("Screen Capture", frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and destroy all windows
out.release()
cv2.destroyAllWindows()

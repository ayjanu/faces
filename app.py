from flask import Flask, request, jsonify
import cv2
from deepface import DeepFace

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the File Processing API!"

@app.route('/photo', methods=['POST'])
def upload_photo():
    # return "Photo!"
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Add your photo processing logic here
    
    return jsonify({'message': 'Photo uploaded and processed successfully'})

@app.route('/video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Add your video processing logic here
    
    return jsonify({'message': 'Video uploaded and processed successfully'})

if __name__ == '__main__':
   app.run(port=5000)

from flask import Flask, render_template, Response

import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Print landmarks
for l in mp_pose.PoseLandmark:
    print(l)

def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle 

cap = cv2.VideoCapture(0)

counter = 0 
stage = None
stage1 = 'up'
stage2 = 'down'
toggle = False
is_webcam_on = False

@app.route('/')
def index():
    return render_template('web.html')

def gen_frames():
    global is_webcam_on
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while is_webcam_on:
            ret, frame = cap.read()
            if not ret:
                break
            print("hi")
            
            white_bg = np.full_like(frame, (255, 255, 255))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            alpha = 0.5
            blended_image = cv2.addWeighted(white_bg, alpha, image, 1 - alpha, 0)
            
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                right_shoulder = []
                right_elbow = []
                right_wrist = []
                angle2 = 0

                if toggle:
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                angle1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                if toggle:
                    angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)

                cv2.putText(blended_image, str(angle1), 
                           tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                if toggle:
                     cv2.putText(blended_image, str(angle2), 
                               tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

                if not toggle:
                    if angle1 > 160:
                        stage = "down"
                    if angle1 < 30 and stage =='down':
                        stage="up"
                        counter +=1
                else:
                    if (angle1 > 160 and angle2 < 30 and stage1 == 'down' and stage2 == 'up'):
                        stage1 = "up"
                        stage2 = "down"
                        counter+=1
                    if angle1 < 30 and angle2 > 160 and stage1 == 'up' and stage2 == 'down':
                        stage1 = "down"
                        stage2 = "up"
                        counter+=1

            except:
                pass

            cv2.rectangle(blended_image, (0,0), (225,73), (245,117,16), -1)
            cv2.putText(blended_image, 'Count', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(blended_image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', cv2.flip(blended_image, 1))
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global is_webcam_on
    is_webcam_on = True
    return 'Webcam started'

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global is_webcam_on
    is_webcam_on = False
    return 'Webcam stopped'

if __name__ == '__main__':
    app.run(debug=True)

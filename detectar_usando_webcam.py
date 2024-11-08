from flask import Flask, jsonify, render_template, Response, request
from ultralytics import YOLO
import cv2

app = Flask(__name__)

model = YOLO("C:/Users/Lucas Colombo/runs/detect/train34/weights/best.pt")

cap = None
is_system_on = False

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/deteccoes', methods=['GET'])
def get_detections():
    global is_system_on, cap
    if not is_system_on:
        return jsonify([])  

    success, img = cap.read()
    if success:
        
        results = model(img)
        detections = []


        for result in results:
            boxes = result.boxes  


            for box in boxes:
                cls = box.cls[0]  
                conf = box.conf[0] 


                class_name = model.names[int(cls)]


                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'box': box.xyxy[0].tolist()
                })

        return jsonify(detections)
    else:
        return jsonify({'error': 'Erro ao capturar a imagem'}), 500

def generate_frames():
    global cap
    while is_system_on:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if not is_system_on:
        return '', 204  
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_system', methods=['POST'])
def toggle_system():
    global is_system_on, cap
    action = request.json.get('action')

    if action == "on":
        if not is_system_on:
            cap = cv2.VideoCapture(0) 
            is_system_on = True
    elif action == "off":
        if is_system_on:
            is_system_on = False
            cap.release() 
            cap = None

    return jsonify({'status': 'on' if is_system_on else 'off'})

if __name__ == '__main__':
    app.run(debug=True)

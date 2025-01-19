from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('path_to_yolov8_model.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files['image'].read()
    results = model(image)
    predictions = results.pandas().xyxy[0].to_dict(orient="records")
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)

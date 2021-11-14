#-*-coding:utf-8 -*-
from flask import Flask
from flask import request, json
import json
from PIL import Image
from download import download_image
from iinit import det


PAGE = '''<!doctype html>
    <title>Object Detection</title>
    <h1>yolo object detection</h1>
    <form action="" method=post enctype=multipart/form-data>
        <p>
         <label for="image">image</label>
         <input type=file name=file required>
         <input type=submit value=detect>
    </form>
    '''

app = Flask('Detection')

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('detector'))


@app.route('/detector', methods=['GET', 'POST'])
def detector():
    if request.method !='POST':
        return PAGE
    img = request.files['file']
    data = img.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    bboxes = detection(img=img)
    bboxes = bboxes.reshape(-1, 8).tolist()
    # print('inference time: ', time.time()-tic)

    return jsonify(msg='success', data={'bboxes': bboxes})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
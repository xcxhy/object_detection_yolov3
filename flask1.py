from flask import Flask,jsonify,render_template,redirect,url_for,request
import os
import sys
import cv2
import base64
# from flask_cors import CORS
import numpy as np
from detection import detection
app = Flask(__name__) #实例化
app.debug =True
app.config['JSON_AS_ASCII'] = False

#设置可跨域范围
#设置可跨域范围
# CORS(app, supports_credentials=True)

@app.route('/',methods=['GET'])

def index():
    # return redirect(url_for('detector'))
    return redirect(url_for('detector'))

@app.route('/detector',methods=['GET','POST'])
def detector():
    if request.method != 'POST':
        return render_template('index.html')
    img = request.files['file']
    img.save('./static/test.jpg')
    detection('static','test.jpg')
    with open('./static/test_result.jpg','rb') as img_f:
        img = img_f.read()
    # data = img.read()
    # arr = np.frombuffer(data,dtype=np.uint8)
    # img = cv2.imdecode(arr,cv2.IMREAD_COLOR)
    img_stream = base64.b64encode(img).decode()
    return render_template('index1.html',img_stream = img_stream)
    
if __name__=='__main__':
    app.run()
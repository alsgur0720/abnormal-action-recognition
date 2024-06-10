# from flask import Flask, render_template, request, jsonify, redirect, url_for
# from main import test

# app = Flask(__name__)


# @app.route('/')
# def home():
#     return render_template('tracking.html')

    
    
# @app.route('/predict', methods=['GET'])
# def predict():
#     print('실행 중 ')
#     prediction = test()
#     print('실행 완료 ')
    
#     # 예측 결과를 JSON 형태로 반환
#     # return jsonify(prediction)
#     return redirect(url_for('home'))


# if __name__ == '__main__':
#     app.run(debug=True)

import os
from flask import Flask, render_template
from flask import Blueprint, send_file, request, redirect, url_for, render_template, session
try:
	from werkzeug.utils import secure_filename
except:
	from werkzeug import secure_filename
 
from main import test
app = Flask(__name__)
app.secret_key = 'super secret key'
UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def index():
    return render_template('index2.html')


@app.route('/action_detection')
def action_detection():
    return render_template('index2_ad.html', video_name=session.get('uploaded_file_name', None))

@app.route('/upload')
def upload():
    return render_template('upload.html')

# 파일 업로드 처리
@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # 저장할 경로 + 파일명
        test()
        f.save(UPLOAD_FOLDER+secure_filename(f.filename))
        session['uploaded_file_name'] = f.filename
        return redirect(url_for('action_detection'))


if __name__ == '__main__':
    app.run(port=8080)

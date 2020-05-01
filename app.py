import numpy as np
from cv2 import GaussianBlur
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import img_to_array, load_img
import pickle as pk
from joblib import dump, load
import datetime
from flask import Flask
from flask import render_template, request, redirect, flash, url_for
import os

def preprocess_input_custom(img1):
    blur = GaussianBlur(img1,(5,5),0)
    img = preprocess_input(blur)
    return img

def predict(image):
    IMG_DIM = (112, 112)
    TEST_SIZE = 1
    Cache_dir = [image]
    X_test = [img_to_array(load_img(file, target_size=IMG_DIM)) for file in Cache_dir]
    vgg16 = load_model('Models/vgg16_finetuned.h5')
    features_test = vgg16.predict(np.array(X_test))
    features_test = np.resize(features_test,(TEST_SIZE,features_test.shape[1]*features_test.shape[2]*features_test.shape[3]))

    mms = pk.load(open("Models/mms.pkl",'rb'))
    x_test = mms.transform(features_test)

    pca = pk.load(open("Models/pca.pkl",'rb'))
    x_test_pca = pca.transform(x_test)

    clf = load('Models/bagging_svc.joblib')

    test_pred = clf.predict(x_test_pca)

    return test_pred[0]


app = Flask(__name__, template_folder='templates')


@app.route('/')
def home_endpoint():
    return redirect('/upload-image')


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            filename = image.filename
            filename = filename.lower()
            jpg = filename.find('jpg')
            jpeg = filename.find('jpeg')
            png = filename.find('png')
            if(jpg==-1 and jpeg==-1 and png==-1):
                flash('Image format should be "png", "jpg" or "jpeg"')
                return redirect(url_for('home_endpoint'))
            answer = predict(image)
            print("Image saved")
            if(answer==0):
                return render_template("0.html")
            else:
                return render_template("1.html")
    return render_template("upload.html")



if __name__ == '__main__':
    # load_model()
    app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
    app.debug = True
    app.config["IMAGE_UPLOADS"] = "Test/class1"
    app.run(threaded=False,debug=False)

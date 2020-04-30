import numpy as np
from cv2 import GaussianBlur
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import pickle as pk
from joblib import dump, load
import datetime
from flask import Flask
from flask import render_template, request, redirect
import os

def preprocess_input_custom(img1):
    blur = GaussianBlur(img1,(5,5),0)
    img = preprocess_input(blur)
    return img

def predict():
    IMG_DIM=112
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_custom)
    test_generator = test_datagen.flow_from_directory(
        './Test',
        target_size=(IMG_DIM, IMG_DIM),
        batch_size=1,
        shuffle=True)
    TEST_SIZE = 1
    X_test = []
    j = 0
    for i in test_generator:
        j+=1
        (a,b) = i
        X_test.append(np.squeeze(a))
        if(j==TEST_SIZE):
            break
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
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], "input.png"))
            answer = predict()
            print("Image saved")
            if(answer==0):
                return render_template("0.html")
            else:
                return render_template("1.html")
    return render_template("upload_image.html")



if __name__ == '__main__':
    # load_model()
    app.debug = True
    app.config["IMAGE_UPLOADS"] = "Test/class1"
    app.run(threaded=False,debug=False)

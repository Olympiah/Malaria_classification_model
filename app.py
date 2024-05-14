from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image

import os
import numpy as np
from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
# import PIL

app = Flask(__name__)
# models/model1.h5
MODEL_PATH ='models/model1.h5'
# loading our saved model
model = load_model(MODEL_PATH)

# defining function
def model_predict(img_path,model):
    img=  image.load_img(img_path,target_size=(224,224))
    # preprocessing image by converting it to an array of pixels
    x = image.img_to_array(img)

    x = np.expand_dims(x,axis=0)

    x = preprocess_input(x)
    preds = model.predict(x)
    # getting the highest probability
    preds = np.argmax(preds,axis=1)

    if preds > 0.5:
        preds = "Parasitized"
    else:
        preds = "Uninfected"
    return preds

# Main Page
@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method =='POST':
        file = request.files['file']

        # saving file to./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(file.filename))
        file.save(file_path)

        result = model_predict(file_path,model)
        return result
    return None


if __name__=='__main__':
    app.run()
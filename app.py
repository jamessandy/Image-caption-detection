import numpy as np
from utils import *
import base64

# # Keras
from keras.preprocessing.image import  img_to_array


# Flask utils
from flask import Flask, url_for, render_template, request,send_from_directory,redirect
from werkzeug.utils import secure_filename
import cv2



app = Flask(__name__)
app.debug = True

@app.route("/")
@app.route('/home')
def index():
    return render_template("index.html")


@app.route('/team')
def team():
    return render_template("team.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route("/generateCaption", methods=["POST"])
def generateCaption():
    image = request.files['image']
    
    imgBinary = image.read()

    # convert string of image data to uint8
    nparr = np.fromstring(imgBinary, np.uint8)
    # decode images
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # text = greedy_search(img)

    # xception_model = Xception(img)
    # xception_caption = xception_model.generate_desc(32)

    imgencoded = cv2.imencode(".jpg", img)[1]

    jpg_as_text = base64.b64encode(imgBinary)

    jpg_as_text = jpg_as_text.decode('ascii')
   
    img = cv2.resize(img, (224, 224))
    # print(jpg_as_text)


    photo = extract_features(img)
    # generate description
    caption = generate_desc(model, tokenizer, photo, max_length)
    # print(f'inception: {text}')
    # print(f'vgg: {caption}')
    # print(f'xception: {xception_caption}')
    # caption = "Cute cat for Erwin Schrodinger"
    return render_template("results.html", image=jpg_as_text, caption=caption)




if __name__ == "__main__":
    app.run()
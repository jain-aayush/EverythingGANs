#importing necessary libraries
import os
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image
from models.ImageSuperResolution import predict as superres_predict
from models.ImageDeraining import predict as derain_predict
from models.AnimateMe import predict as animate_predict

CURRENT_WORKING_DIRECTORY = Path(os.getcwd())
UPLOAD_FOLDER = CURRENT_WORKING_DIRECTORY/'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}#the image extensions that are accepted

#loading environment variables
load_dotenv(CURRENT_WORKING_DIRECTORY/'variables.env')

#configuring the flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.getenv('secretKey')

#home route
@app.route("/")
def home():
    return render_template('home.html') 

#about route
@app.route("/about")
def about():
    return render_template('about.html')

#helper functiont to find in the file has one of the allowed extensions
def allowed_file(filename):
    extension = filename.split('.')[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False
    else:
        return True

#superresolution route
@app.route("/superresolution", methods = ['GET', 'POST'])
def superresolution():
    if(request.method == 'POST'):
        if('file' not in request.files):
            flash('No File')
            return redirect(request.url)
        file = request.files['file']

        if(file.filename == ''):
            flash('No file selected')
            return redirect(request.url)
        if(file and allowed_file(file.filename)):
            filename = secure_filename(file.filename)
            file.save(UPLOAD_FOLDER/filename)
            image_file = url_for('static', filename = 'uploads/'+filename)
            return render_template('superresolution.html', image_file = image_file)
    return render_template('superresolution.html', image_file = None)

#route for superresolve
@app.route("/superresolve<image_file>/gans", methods = ['GET', 'POST'])
def superresolve(image_file):
    image = Image.open(CURRENT_WORKING_DIRECTORY/'static/uploads'/image_file)
    predicted_image = superres_predict(image)
    predicted_image = np.uint8(predicted_image)#converting the dtype of the predicted image to be allowed to load using  PIL
    predicted_image = Image.fromarray(predicted_image)
    new_name = image_file.split('.')[0] + '_superresolved.jpeg'
    predicted_image.save(CURRENT_WORKING_DIRECTORY/'static/uploads'/new_name)
    image_file = url_for('static', filename = 'uploads/' + new_name)
    return render_template('superresolution.html', image_file = image_file)

#deraining route
@app.route("/deraining", methods = ['GET', 'POST'])
def deraining():
    if(request.method == 'POST'):
        if('file' not in request.files):
            flash('No File')
            return redirect(request.url)
        file = request.files['file']

        if(file.filename == ''):
            flash('No file selected')
            return redirect(request.url)
        if(file and allowed_file(file.filename)):
            filename = secure_filename(file.filename)
            file.save(UPLOAD_FOLDER/filename)
            image_file = url_for('static', filename = 'uploads/'+filename)
            return render_template('derain.html', image_file = image_file)
    return render_template('derain.html', image_file = None)

#route for derain
@app.route("/derain<image_file>/gans", methods = ['GET', 'POST'])
def derain(image_file):
    image = Image.open(CURRENT_WORKING_DIRECTORY/'static/uploads'/image_file)
    predicted_image = derain_predict(image)
    predicted_image = np.uint8((predicted_image*0.5 + 0.5)*255.0)#re-normalizing the predicted image pixels to be in [0,1]
    predicted_image = Image.fromarray(predicted_image)
    new_name = image_file.split('.')[0] + '_derained.jpeg'
    predicted_image.save(CURRENT_WORKING_DIRECTORY/'static/uploads'/new_name)
    image_file = url_for('static', filename = 'uploads/' + new_name)
    return render_template('derain.html', image_file = image_file)

#animateme route
@app.route("/animateme", methods = ['GET', 'POST'])
def animateme():
    if(request.method == 'POST'):
        if('file' not in request.files):
            flash('No File')
            return redirect(request.url)
        file = request.files['file']

        if(file.filename == ''):
            flash('No file selected')
            return redirect(request.url)
        if(file and allowed_file(file.filename)):
            filename = secure_filename(file.filename)
            file.save(UPLOAD_FOLDER/filename)
            image_file = url_for('static', filename = 'uploads/'+filename)
            return render_template('animate.html', image_file = image_file)
    return render_template('animate.html', image_file = None)

#route for animate
@app.route("/animate<image_file>/gans", methods = ['GET', 'POST'])
def animate(image_file):
    image = Image.open(CURRENT_WORKING_DIRECTORY/'static/uploads'/image_file)
    predicted_image = animate_predict(image)
    predicted_image = np.uint8((predicted_image*0.5 + 0.5)*255.0)#re-normalizing the predicted image pixels to be in [0,1]
    predicted_image = Image.fromarray(predicted_image)
    new_name = image_file.split('.')[0] + '_animated.jpeg'
    predicted_image.save(CURRENT_WORKING_DIRECTORY/'static/uploads'/new_name)
    image_file = url_for('static', filename = 'uploads/' + new_name)
    return render_template('animate.html', image_file = image_file)



if __name__ == '__main__':
    app.run(debug = True)

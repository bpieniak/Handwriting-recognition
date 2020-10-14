import flask
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from cv2 import cv2
import base64
import string

app = flask.Flask(__name__, template_folder='templates')
model = keras.models.load_model('./Models/acc98')
alphabet = list(string.ascii_uppercase)

@app.route('/')
def home():
	return flask.render_template('draw.html',  prediction = None)


@app.route('/', methods=['POST'])
def predict():
	draw = flask.request.form['url']
	draw = draw[22:]
	draw_decoded = base64.b64decode(draw)
	
	image = np.asarray(bytearray(draw_decoded))
	image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
	resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
	
	vect = np.asarray(resized, dtype="uint8")
	vect = vect.reshape(1, 28, 28).astype('float32')
	
	pred = model.predict(vect)
	index_pred = np.argmax(pred)
	print(index_pred)

	return flask.render_template('draw.html', prediction = alphabet[index_pred])

if __name__ == '__main__':
	app.run(debug=True)
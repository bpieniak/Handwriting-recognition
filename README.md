# Handwriting recognition
Recognition of handwritten characters with convolutional neural network using tensorflow 2.3. Used datasets: [A-Z Handwritten Alphabets](https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format), [MNIST](http://yann.lecun.com/exdb/mnist/). My motivation to make this project was to acquire some skills in deploying machine learning models so that anyone can use it. 

## Live demo available [here](https://hw-recognition-cnn.herokuapp.com/).


To use it just draw any digit or latin alphabet letter and press Predict button.

![Demo](https://github.com/pt3k/Handwriting-recognition/blob/master/demo.gif)

This site is build using flask + bootstrap. Drawing part is made using HTML5 Canvas and JavaScript.

This project use convolutional neural network written using Tensorflow to make predictions. It was trained on combination of MNIST and A-Z Handwritten Alphabets. For training I used simple augmentation (rotation and scaling). Training notebook is available here.


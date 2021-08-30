from flask import Flask,jsonify,request
#from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from PIL import Image
import numpy as np

def simple_model():
  
  img_input = layers.Input(shape=(224, 224, 3))
  x = layers.Conv2D(16, 3, activation='relu')(img_input)
  x = layers.MaxPooling2D(2)(x)
  x = layers.Conv2D(32, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  x = layers.Conv2D(64, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  x = layers.Conv2D(128, 3, activation='relu')(x)
  x = layers.MaxPooling2D(2)(x)
  x = layers.Flatten()(x)
  x = layers.Dropout(0.25)(x)
  x = layers.Dense(512, activation='relu')(x)
  output = layers.Dense(3, activation='softmax')(x)

  model = Model(img_input, output)
  model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(lr=0.001),
              metrics=['acc'])


  return model


app = Flask(__name__)
#model = tf.keras.models.load_model("final_model.h5")
#print(model.summary())

model = simple_model()
#tf.keras.utils.plot_model(model, show_shapes=False)
model.load_weights('final_model.h5')

print("Model Loaded Successfully")
classes = ['Other','Potato_Late_blight','Potato_healthy']

@app.route('/',methods=['GET'])
def test():
    return "<h1>Agritech Crop Doctor Server is Up and Running</h1> \n POST a request on \"https://crop-doctor-namal.herokuapp.com/predict\" for prediction"

@app.route('/predict',methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'File not recieved'
    else:
        img_file = Image.open(request.files['image'])
        img = img_file.resize([224,224])
        img = np.array(img)
        img = tf.cast(img,tf.float32)
        return predict(img)

def predict(img):
    img = np.expand_dims(img,axis=0)
    result = model.predict(img)
    ind = np.argmax(result)
    disease_name = classes[ind]
    print("prediction from model: "+disease_name)
    return disease_name

if __name__ == '__main__':
    #app.run(host="0.0.0.0", port=5000, debug=False,threaded=False)
    app.run()

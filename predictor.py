import keras
import pandas as pd
from utils import *
from sklearn.preprocessing import StandardScaler, LabelEncoder

newmodel = keras.models.load_model('LSTM_model.h5')
path = '/home/srushti/Documents/Old Ubuntu Data/Project/emotion_files/1.wav'
s = get_features(path)
r = s[0:1, :]
result = np.expand_dims(r, axis=2)
prediction = newmodel.predict(result)
print(prediction)

Ft = pd.read_csv('features1.csv')
Y = Ft['labels'].values
Y.shape
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)

predicted_class = np.argmax(prediction, axis=1)
#Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
re = encoder.inverse_transform(predicted_class)
print(re)



import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()   

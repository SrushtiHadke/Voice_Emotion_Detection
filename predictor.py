import keras
import pandas as pd
from utils import *
from sklearn.preprocessing import StandardScaler, OneHotEncoder

newmodel = keras.models.load_model('srushti.keras')
path = '/home/srushti/Documents/Project/Librosa/Voice Emotion Detection/emotions_seperated/sad/03-01-04-01-01-02-19.wav'
s = get_features(path)
r = s[0:1, :]
result = np.expand_dims(r, axis=2)
prediction = newmodel.predict(result)
print(prediction)

Features = pd.read_csv('features1.csv')
Y = Features['labels'].values
Y.shape
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
re = encoder.inverse_transform(prediction)
print(re)

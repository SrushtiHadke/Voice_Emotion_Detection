import gradio as gr
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import get_features

# Load the trained model
newmodel = keras.models.load_model('LSTM_model.h5')

# Load labels and encode them
Ft = pd.read_csv('features1.csv')
Y = Ft['labels'].values
encoder = LabelEncoder()
encoder.fit(Y)

# Function to predict emotion from an audio file
def predict_emotion(audio_file):
    try:
        # Extract features from the uploaded audio file
        features = get_features(audio_file)
        features = features[0:1, :]  # Adjust shape if necessary
        features_expanded = np.expand_dims(features, axis=2)

        # Make prediction
        prediction = newmodel.predict(features_expanded)

        # Get predicted label
        predicted_class = np.argmax(prediction, axis=1)
        predicted_label = encoder.inverse_transform(predicted_class)

        return f"Predicted Emotion: {predicted_label[0]}"
    except Exception as e:
        return f"Error: {str(e)}"

# Define Gradio interface
inputs = gr.Audio(type="filepath", label="Upload an Audio File")
outputs = gr.Textbox(label="Emotion Prediction")

demo = gr.Interface(
    fn=predict_emotion,
    inputs=inputs,
    outputs=outputs,
    title="Voice Emotion Detection",
    description="Upload an audio file to predict the emotional tone."
)

# Launch the interface
if __name__ == "__main__":
    demo.launch()

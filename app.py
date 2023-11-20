from flask import Flask, render_template, request, jsonify
from scipy.io.wavfile import write
import io
import librosa
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load your SER model
model = load_model('SER_model.h5')  # Replace with the path to your model file

# Define the predict_emotion function
def predict_emotion(audio_file):
    # Extract MFCC features from the audio file
    mfcc = extract_mfcc(audio_file)  # Assuming you've defined this function

    # Reshape the MFCC features to match the model's input shape
    mfcc = mfcc.reshape(1, 40, 1)  # Shape should match (batch_size, num_features, num_timesteps)

    # Make predictions using the loaded model
    predictions = model.predict(mfcc)

    # Get the predicted emotion label
    predicted_label = np.argmax(predictions)

    if predicted_label==0:
        return 'Emotion: Angry'
    elif predicted_label==1:
        return 'Emotion: Disgust'
    elif predicted_label==2:
        return 'Emotion: Fear'
    elif predicted_label==3:
        return 'Emotion: Happy'
    elif predicted_label==4:
        return 'Emotion: Neutral'
    elif predicted_label==5:
        return 'Emotion: Pleasant_surprise'
    else:
        return 'Emotion: Sad'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_emotion", methods=["POST"])
def predict_emotion_route():
    if "audio" in request.files:
        audio_data = request.files["audio"].read()
        emotion_index = predict_emotion(audio_data)

        # Replace with your emotion labels
        emotion_labels = ["Emotion1", "Emotion2", "Emotion3"]  # Add your labels here

        predicted_emotion = emotion_labels[emotion_index]

        return jsonify({"emotion": predicted_emotion})

if __name__ == "__main__":
    app.run(debug=True)

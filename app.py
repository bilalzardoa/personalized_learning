import keras
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
model_path = "my_model.keras"
model = load_model(model_path) 
import keras
import tensorflow as tf

def tokenize(text):
    tokenizer = Tokenizer()  # No limit on the number of words
    tokenizer.fit_on_texts(text)  # Fit the tokenizer on the input text
    sequences = tokenizer.texts_to_sequences(text)  # Convert text to sequences
    word_index = tokenizer.word_index  # Get the word index

    return word_index, sequences

def one_hot_encode(sequences, dimension):
    one_hot_encoded_text = np.zeros((len(sequences), dimension))  # Initialize the one-hot array

    # Loop through each sequence to fill in the one-hot encoding
    for row_idx, seq in enumerate(sequences):
        for word_index in seq:  # For each word index in the sequence
            if word_index > 0:  # Ensure word index is valid
                one_hot_encoded_text[row_idx, word_index - 1] = 1  # Set corresponding index to 1

    return one_hot_encoded_text


def pad(data,max_length):
  padded_data = pad_sequences(data, maxlen=max_length, padding='post')

  return padded_data

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This line enables CORS

def preprocess_text(text):
   
    word_index, sequences = tokenize([text])  
    processed_text = one_hot_encode(sequences, dimension=len(word_index) + 1)
    padded_text = pad(processed_text, 18871)  
    return np.array(padded_text.reshape(1, -1))

 
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text data from the POST request
    data = request.get_json(force=True)
    text = data.get('text')
    # Preprocess the text data
    processed_text = preprocess_text(text)

    # Make predictions using the model
    prediction = model.predict(processed_text)

    # Assuming prediction is a 2D array, e.g., shape (1, 6)
    y_pred = prediction  # or any other transformation needed based on your model output

    # Create evaluation and sorted evaluation list
    labels = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    evaluation = dict()

    for i in range(y_pred.shape[1]):
        evaluation[labels[i]] = y_pred[0][i]

    # Sort the evaluation dictionary
    sorted_evaluation = dict(sorted(evaluation.items(), key=lambda item: item[1]))

    # Create a sorted list of tuples (label, score)
    sorted_list = [(label, float(score)) for label, score in sorted_evaluation.items()]

    # Return the sorted prediction as JSON
    return jsonify({'sorted_prediction': sorted_list})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
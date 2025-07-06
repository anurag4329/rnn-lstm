# RNN-LSTM
🧠 Next Word Prediction Using LSTM
📌 Overview
This project demonstrates a deep learning approach to predict the next word in a given sequence using an LSTM (Long Short-Term Memory) model. It uses the rich and complex language of Shakespeare's Hamlet as training data, making it a compelling example of sequence modeling in natural language processing (NLP).

🔍 Key Features
Sequence prediction using LSTM-based neural networks

Trained on Shakespeare’s Hamlet text

Interactive prediction via a Streamlit web app

🧪 Project Pipeline
1. Data Collection
The dataset is sourced from Shakespeare's Hamlet, providing a diverse and challenging vocabulary for the model to learn from.

2. Data Preprocessing
The raw text is cleaned and tokenized.

Word sequences are created and padded to ensure consistent input lengths.

Data is split into training and testing sets.

3. Model Architecture
The model is built using the following layers:

🔤 Embedding Layer: Converts each word into a 100-dimensional vector representation.

🧠 LSTM Layer (150 units): First LSTM layer with return_sequences=True to allow the next LSTM to process the entire sequence.

🔁 Dropout Layer (rate = 0.2): Reduces overfitting by randomly disabling 20% of neurons during training.

🧠 LSTM Layer (100 units): Processes the output of the previous LSTM layer for sequential understanding.

🎯 Dense Output Layer: Applies a softmax activation to output the probability distribution over the vocabulary for the next word prediction.

4. Model Training
The model is trained on the processed word sequences.

Early stopping is used to monitor the validation loss and halt training when no further improvements are observed.

5. Evaluation
The model is tested on manually crafted sequences to evaluate its ability to accurately predict the next word in context.

6. Deployment
A Streamlit web application allows users to:

Enter a phrase

Receive the predicted next word in real time

🚀 Try It Yourself
Clone the repo and run the Streamlit app locally:

bash
Copy
Edit
streamlit run app.py
🛠 Technologies Used

Python 🐍

TensorFlow / Keras 🧠

Streamlit 🌐

NLTK for preprocessing 📚


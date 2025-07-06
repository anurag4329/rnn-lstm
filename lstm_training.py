import nltk
import numpy as np
import tensorflow as tf
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
# nltk.download('gutenberg')
from nltk.corpus import gutenberg
import  pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout


data=gutenberg.raw('shakespeare-hamlet.txt')
## save to a file
with open('hamlet.txt','w') as file:
    file.write(data)

##laod the dataset
with open('hamlet.txt','r') as file:
    text=file.read().lower()

## Tokenize the text-creating indexes for words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1


## create input sequences

input_sequences = []

for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

## Padding the sequences

max_sequence_len = max([len(x) for x in input_sequences])
print(max_sequence_len)

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

x,y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

## split the training-test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("Vocabulary size:", total_words)
print("Max sequence length:", max_sequence_len)


## define the LSTM model

## Define the model
model=Sequential()
model.add(Embedding(total_words,100,input_length=max_sequence_len-1))
model.add(LSTM(150,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words,activation="softmax"))

# #Compile the model
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
model.build(input_shape=(None, max_sequence_len))

model.summary()

##Train the model
history=model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),verbose=1)

## Save the model
model.save("next_word_lstm.h5")
## Save the tokenizer
import pickle
with open('tokenizer.pickle','wb') as handle:
    pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)

from tensorflow.keras.models import load_model
model = load_model("next_word_lstm.h5")

#
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)


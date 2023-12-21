import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

train_data_path = 'train1.csv'  
test_data_path = 'test1.csv'    
val_data_path = 'test_validation.csv' 

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)
val_data = pd.read_csv(val_data_path)

max_words = 10000  

# Initialize the tokenizer
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data['job description'])

# Convert the text in each dataset to sequences
train_sequences = tokenizer.texts_to_sequences(train_data['job description'])
test_sequences = tokenizer.texts_to_sequences(test_data['job description'])
val_sequences = tokenizer.texts_to_sequences(val_data['job description'])

max_len = 100  

train_labels = np.asarray(train_data['label'].astype('float32'))
val_labels = np.asarray(val_data['label'].astype('float32'))

# Pad sequences
padded_train_data = pad_sequences(train_sequences, maxlen=max_len)
padded_test_data = pad_sequences(test_sequences, maxlen=max_len)
padded_val_data = pad_sequences(val_sequences, maxlen=max_len)


model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=32, input_length=max_len)) # Embedding layer
model.add(LSTM(32))  # LSTM layer with 32 units
model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(padded_train_data, train_labels, epochs=100, batch_size=64, validation_data=(padded_val_data, val_labels))

# Load test data
test_data_path = 'test1.csv'  # Replace with your file path
test_data = pd.read_csv(test_data_path)

# Tokenize and pad the test data
test_sequences = tokenizer.texts_to_sequences(test_data['job description'])
padded_test_data = pad_sequences(test_sequences, maxlen=max_len)

predicted_labels = model.predict(padded_test_data)
predicted_labels = (predicted_labels > 0.5).astype('int32')  # Convert probabilities to 0 or 1

val_data_path = 'test_validation.csv'
validation_data = pd.read_csv(val_data_path)
actual_labels = np.asarray(validation_data['label'].astype('int32'))

# Calculate metrics
accuracy = accuracy_score(val_labels, predicted_labels)
precision = precision_score(val_labels, predicted_labels)
recall = recall_score(val_labels, predicted_labels)
f1 = f1_score(val_labels, predicted_labels)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

def predict_job_offer(model, tokenizer, job_offer_text, max_len=100):
    sequence = tokenizer.texts_to_sequences([job_offer_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    predicted_label = (prediction > 0.5).astype('int32')

    # Interpret the prediction
    if predicted_label[0][0] == 1:
        return "Dangerous/Scam"
    else:
        return "Safe"


new_job_offer = ""

result = predict_job_offer(model, tokenizer, new_job_offer, max_len)
print(f"The job offer is predicted to be: {result}")


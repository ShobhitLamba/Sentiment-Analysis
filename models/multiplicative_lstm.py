# Recurrent Neural Network with Multiplicative-LSTM running over imdb dataset
# Author: Shobhit Lamba
# e-mail: shobhit.lamba@uic.edu

# Importing the libraries
from keras.models import Sequential
from keras.layers import Embedding, Dense
from keras.preprocessing import sequence
from keras.datasets import imdb
from sklearn.metrics import precision_recall_fscore_support as score
from utils.multiplicative_LSTM import MultiplicativeLSTM

MAX_FEATURES = 20000
batch_size = 32
MAX_SEQUENCE_LENGTH = 80

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = MAX_FEATURES)

x_train = sequence.pad_sequences(x_train, MAX_SEQUENCE_LENGTH)
x_test = sequence.pad_sequences(x_test, MAX_SEQUENCE_LENGTH)

# Building the network architecture
model = Sequential()
model.add(Embedding(MAX_FEATURES, 128))
model.add(MultiplicativeLSTM(128, dropout = 0.2, recurrent_dropout = 0.2)) 
model.add(Dense(1, activation = "sigmoid"))

# Compiling the network
model.compile(loss = "binary_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

model.summary()

# Training
model.fit(x_train, y_train, 
          batch_size = batch_size, 
          epochs = 10,
          validation_data = (x_test, y_test))

# Evaluating results
predicted_result = model.predict_classes(x_test, batch_size = batch_size)

print("\n\n_________________________\nResult", y_test, '\n_________________________\n\n')
    
precision, recall, fscore, support = score(y_test, predicted_result)
count = 0
for i in range(len(y_test)):
    if(y_test[i] == predicted_result[i]):
        count+=1

print('accuracy: ', count/len(y_test))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


import numpy as np
import random
#Generate random numbers between 0 and 15 for training data
symbols = np.random.randint(0,16,100000)
X_train = np.zeros((symbols.size, symbols.max()+1))
X_train[np.arange(symbols.size),symbols] = 1
#X_train.shape


#Generate random numbers between 0 and 15 for test data
symbols = np.random.randint(0,16,10000)
X_test = np.zeros((symbols.size, symbols.max()+1))
X_test[np.arange(symbols.size),symbols] = 1



import tensorflow as tf
from tensorflow import keras
from keras.models import Model 
from keras.layers import Input, Dense, ReLU, BatchNormalization 
inputs= X_test.shape[1]

#define the encoder 
encoder_input= Input(shape=(inputs,))
encoder_layer=Dense(inputs, activation="relu")(encoder_input)
encoder_layer=Dense(7, activation="linear")(encoder_layer)
encoder_output= BatchNormalization()(encoder_layer) 

R=4/7;
Eb_No_training=7; ## dB
Eb_No_training_ratio=10**(Eb_No_training/10);
stddev=(2*R*Eb_No_training_ratio)**(-0.5);
channel_output=tf.keras.layers.GaussianNoise(stddev)(encoder_output)

# define the decoder 
decoder= Dense(inputs, activation="relu")(channel_output) 
decoder_output = Dense(inputs, activation="softmax")(decoder)

# define the autoencoder

autoencoder=Model(inputs=encoder_input, outputs=decoder_output)

## encoder side
encoder=Model(inputs=encoder_input, outputs=encoder_output)

#decoder side 
decoder=Model(inputs=channel_output, outputs=decoder_output)

autoencoder.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
history=autoencoder.fit(X_train,X_train,epochs=10, validation_data=(X_test,X_test),batch_size=128)
#autoencoder.save("functional_encoder.h5")


autoencoder.summary()



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt



#Generate random numbers between 0 and 15 for BER calculations
test_size=1000000;
symbols = np.random.randint(0,16,test_size)
X_test = np.zeros((symbols.size, symbols.max()+1))
X_test[np.arange(symbols.size),symbols] = 1

#generating a E/N vector
R=4/7;
num_steps=26
Eb_No=np.linspace(-4,8,num_steps); ## dB


BER=np.zeros(num_steps)
for i in range(len(Eb_No)):
  Eb_No_ratio=10**(Eb_No[i]/10);
  stddev=(2*R*Eb_No_ratio)**(-0.5);
  encodedsymbols=encoder.predict(X_test)
  noise = np.random.normal(0,stddev,[test_size,7])
  noisy=encodedsymbols+noise
  a=decoder.predict(noisy)
  idx = np.argmax(a, axis=-1)
  a = np.zeros( a.shape )
  a[ np.arange(a.shape[0]), idx] = 1
  recovered_symbols=np.argmax(a, axis=1)
  errors=np.count_nonzero(recovered_symbols-symbols)
  BER[i]=errors/len(symbols)


plt.figure()
plt.semilogy(Eb_No, BER, 'k--')
plt.title('Block Error Rate')
plt.xlabel('E_b/N_o')
plt.grid(True)
epochs = range(1, len(acc)+1)
plt.figure()
plt.plot(epochs, acc, 'k--', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'k--', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


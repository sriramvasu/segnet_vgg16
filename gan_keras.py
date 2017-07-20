from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sp


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_train = (X_train - 127.5) / 127.5

generator=Sequential()
generator.add(Dense(128*7*7,input_dim=100,activation=LeakyReLU(0.2)))
generator.add(BatchNormalization())
generator.add(Reshape((7,7,128)))
generator.add(UpSampling2D())
generator.add(Convolution2D(64,5,5,border_mode='same',activation=LeakyReLU(0.2)))
generator.add(BatchNormalization())
generator.add(UpSampling2D())
generator.add(Convolution2D(1,5,5,border_mode='same',activation='tanh'))


discriminator=Sequential()
discriminator.add(Convolution2D(64,5,5,subsample=(2,2),input_shape=(28,28,1),border_mode='same',activation=LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Convolution2D(128,5,5,subsample=(2,2),border_mode='same',activation=LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1,activation='sigmoid'))


generator.summary()
discriminator.summary()

generator.compile(loss='binary_crossentropy',optimizer=Adam())
discriminator.compile(loss='binary_crossentropy',optimizer=Adam())

gan_input=Input(shape=[100])
x=generator(gan_input)
gan_output=discriminator(x)
gan=Model(input=gan_input,output=gan_output)
gan.compile(loss='binary_crossentropy',optimizer=Adam())
gan.summary()


def train(n_epochs,batch_size):
	n_batches=X_train.shape[0]//batch_size
	for i in range(n_epochs):
		for j in range(n_batches):
			print 'epoch:',i,'batch:',j
			noise_input=np.random.standard_normal([batch_size,100])
			data_input=X_train[np.random.randint(0,X_train.shape[0],size=batch_size)]
			generations=generator.predict(noise_input)
			X=np.concatenate([generations,data_input])
			Y=np.concatenate([np.zeros(batch_size),np.ones(batch_size)])
			discriminator.trainable=True
			discriminator.train_on_batch(X,Y)


			noise_input=np.random.standard_normal([batch_size,100])
			y_gen=np.ones(batch_size)
			discriminator.trainable=False
			gan.train_on_batch(noise_input,y_gen)

# train(50, 128)
# generator.save_weights('gen_50_scaled_images.h5')
# discriminator.save_weights('dis_50_scaled_images.h5')
generator.load_weights('gen_50_scaled_images.h5')
discriminator.load_weights('dis_50_scaled_images.h5')
out=generator.predict(np.random.standard_normal([100,100]))
print out.shape

plt.figure(1)
for i in range(out.shape[0]):
	plt.subplot(10,10,i+1)
	plt.imshow(out[i,:].reshape([28,28]),'gray')
	plt.axis('off')
plt.show()
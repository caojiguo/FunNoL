print("This python script requires libraries: tensorflow, numpy, sklearn")
print("To run the script: python starlight_compile.py StarLightCurves_TRAIN.tsv StarLightCurves_TEST.tsv")

#Load dependencies
import tensorflow.keras as keras
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed,Bidirectional, Dropout
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import sys
import os 




#Read in data from command line input 
traindata = sys.argv[1]
testdata  = sys.argv[2]

# If you need to run step by step:
# train_data = np.genfromtxt('StarLightCurves_TRAIN.tsv',delimiter = '\t')
# test_data = np.genfromtxt('StarLightCurves_Test.tsv', delimiter = '\t')


#Some settings    
train_or_new = input('1:Start training a new one \n 2: Load the trained model \n')
cpu_or_gpu   = input('a:Train/Apply model with GPU \n b:Train/Apply model with CPU \n')
numofepochs  = input('number of epochs (if required):\n')
train_data = np.genfromtxt(sys.argv[1],delimiter = '\t')
test_data  = np.genfromtxt(sys.argv[2], delimiter = '\t')


#Data features for Starlight curves
timesteps = train_data.shape[1]-1
input_dim = 1 
N = train_data.shape[0]

train_Y = train_data[:,0].reshape(N,1)[0:8192,:]
train_temp =  train_data[:,1:(timesteps+1)].reshape(N, timesteps, input_dim)[0:8192,:]

def downsampling(row):
    return np.mean(row.reshape(-1, 2), axis=1)
train_temp = np.apply_along_axis(downsampling, 1, train_temp)

train_X_un   = np.array(train_temp)
train_X      = np.array(train_temp)
train_X_ori  = np.array(train_temp)

N = 8192

test_Y  = test_data[:,0].reshape(test_data.shape[0],1)
test_temp  = test_data[:,1:(timesteps+1)].reshape(test_data.shape[0],timesteps,input_dim)
test_temp = np.apply_along_axis(downsampling, 1, test_temp)
test_X_un  = np.array(test_temp)
test_X     = np.array(test_temp)
test_X_ori = np.array(test_temp)

timesteps = int(timesteps /2)

train_X =  train_X + np.random.normal(loc = 0, scale = np.sqrt(0.01), size = train_X.shape)
test_X = test_X + np.random.normal(loc = 0 , scale = np.sqrt(0.01), size = test_X.shape)


if train_or_new == '1':

    if cpu_or_gpu == 'a':
        
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')

        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

        #We have used a simple network for illustrations and fast training
        batch_size = 32
        latent_dim = 16
        input_seq = keras.Input(shape = (timesteps, input_dim))
        encoder_layer = Bidirectional(LSTM(32, return_sequences = True,batch_input_shape=(batch_size, timesteps, 1)))(input_seq)
        encoder_layer = Bidirectional(LSTM(latent_dim, return_sequences = False),name='encoder_layer')(encoder_layer)
        decoder_layer = RepeatVector(timesteps)(encoder_layer)
        decoder_layer = Bidirectional(LSTM(latent_dim, return_sequences = True))(decoder_layer)
        decoder_layer = TimeDistributed(Dense(input_dim), name = 'decoder_output')(decoder_layer)

        x = Dense(512)(encoder_layer)
        x = Dropout(0.3)(x)
        x = Dense(256)(x)
        x = Dropout(0.4)(x)
        x = Dense(128)(x)
        x = Dropout(0.4)(x)
        x = Dense(128)(x)
        x = Dropout(0.4)(x)
        x = Dense(128)(x)
        x = Dropout(0.4)(x)
        x = Dense(64)(x)
        x = Dropout(0.3)(x)
        x = Dense(3, activation = 'sigmoid', name = 'class_output')(x)

        model = keras.Model(inputs = input_seq, outputs = [x, decoder_layer])
        losses = {
            "class_output": "categorical_crossentropy",
            "decoder_output": 'mse',
        }
        model.compile(optimizer = 'adam',loss = losses, loss_weights={'class_output': 1,'decoder_output': 1} )
        ohe = OneHotEncoder()
        train_Y_ohe = ohe.fit_transform(train_Y).toarray()
        test_Y_ohe  = ohe.fit_transform(test_Y).toarray()

        history = model.fit(train_X, {'class_output':train_Y_ohe, 'decoder_output':train_X_un}, epochs = int(numofepochs), batch_size = batch_size, validation_data= (test_X,{'class_output': test_Y_ohe, 'decoder_output': test_X_un}))

        encoder = keras.Model(input_seq, encoder_layer)
        encoded_train = encoder(train_X).numpy()
        encoded_test = encoder(test_X).numpy()  

        #Logistic classifer based on predicted representation 
        logistic_regression= LogisticRegression()
        logistic_regression.fit(encoded_train,train_Y.reshape(-1))
        y_pred = logistic_regression.predict(encoded_test)
        print('Accuracy: ', metrics.accuracy_score(test_Y, y_pred))

    if cpu_or_gpu == 'b':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        batch_size = 32
        latent_dim = 16
        input_seq = keras.Input(shape = (timesteps, input_dim))
        encoder_layer = Bidirectional(LSTM(32, return_sequences = True,batch_input_shape=(batch_size, timesteps, 1)))(input_seq)
        encoder_layer = Bidirectional(LSTM(latent_dim, return_sequences = False),name='encoder_layer')(encoder_layer)
        decoder_layer = RepeatVector(timesteps)(encoder_layer)
        decoder_layer = Bidirectional(LSTM(latent_dim, return_sequences = True))(decoder_layer)
        decoder_layer = TimeDistributed(Dense(input_dim), name = 'decoder_output')(decoder_layer)

        x = Dense(512)(encoder_layer)
        x = Dropout(0.3)(x)
        x = Dense(256)(x)
        x = Dropout(0.4)(x)
        x = Dense(128)(x)
        x = Dropout(0.4)(x)
        x = Dense(128)(x)
        x = Dropout(0.4)(x)
        x = Dense(128)(x)
        x = Dropout(0.4)(x)
        x = Dense(64)(x)
        x = Dropout(0.3)(x)
        x = Dense(3, activation = 'sigmoid', name = 'class_output')(x)

        model = keras.Model(inputs = input_seq, outputs = [x, decoder_layer])
        losses = {
            "class_output": "categorical_crossentropy",
            "decoder_output": 'mse',
        }
        model.compile(optimizer = 'adam',loss = losses, loss_weights={'class_output': 1,'decoder_output': 1} )
        ohe = OneHotEncoder()
        train_Y_ohe = ohe.fit_transform(train_Y).toarray()
        test_Y_ohe  = ohe.fit_transform(test_Y).toarray()


        history = model.fit(train_X, {'class_output':train_Y_ohe, 'decoder_output':train_X_un}, epochs = int(numofepochs), batch_size = batch_size, validation_data= (test_X,{'class_output': test_Y_ohe, 'decoder_output': test_X_un}))

        encoder = keras.Model(input_seq, encoder_layer)
        encoded_train = encoder(train_X).numpy()
        encoded_test = encoder(test_X).numpy()  


        logistic_regression= LogisticRegression()
        logistic_regression.fit(encoded_train,train_Y.reshape(-1))
        y_pred = logistic_regression.predict(encoded_test)
        print('Accuracy: ', metrics.accuracy_score(test_Y, y_pred))

elif train_or_new == '2':     
        if cpu_or_gpu =='b':
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        batch_size = 32
        latent_dim = 16
        input_seq = keras.Input(shape = (timesteps, input_dim))
        encoder_layer = Bidirectional(LSTM(32, return_sequences = True,batch_input_shape=(batch_size, timesteps, 1)))(input_seq)
        encoder_layer = Bidirectional(LSTM(latent_dim, return_sequences = False),name='encoder_layer')(encoder_layer)
        decoder_layer = RepeatVector(timesteps)(encoder_layer)
        decoder_layer = Bidirectional(LSTM(latent_dim, return_sequences = True))(decoder_layer)
        decoder_layer = TimeDistributed(Dense(input_dim), name = 'decoder_output')(decoder_layer)

        x = Dense(512)(encoder_layer)
        x = Dropout(0.3)(x)
        x = Dense(256)(x)
        x = Dropout(0.4)(x)
        x = Dense(128)(x)
        x = Dropout(0.4)(x)
        x = Dense(128)(x)
        x = Dropout(0.4)(x)
        x = Dense(128)(x)
        x = Dropout(0.4)(x)
        x = Dense(64)(x)
        x = Dropout(0.3)(x)
        x = Dense(3, activation = 'sigmoid', name = 'class_output')(x)

        model = keras.Model(inputs = input_seq, outputs = [x, decoder_layer])
        losses = {
            "class_output": "categorical_crossentropy",
            "decoder_output": 'mse',
        }
        model.compile(optimizer = 'adam',loss = losses, loss_weights={'class_output': 1,'decoder_output': 1} )
        ohe = OneHotEncoder()
        train_Y_ohe = ohe.fit_transform(train_Y).toarray()
        test_Y_ohe  = ohe.fit_transform(test_Y).toarray()

        funnol = keras.models.load_model('Starlight_model_compile.h5')
        encoder = keras.Model(funnol.input, funnol.get_layer('encoder_layer').output)
        
        encoded_train = encoder(train_X).numpy()
        encoded_test = encoder(test_X).numpy()  


        logistic_regression= LogisticRegression()
        logistic_regression.fit(encoded_train,train_Y.reshape(-1))
        y_pred = logistic_regression.predict(encoded_test)
        print('Accuracy: ', metrics.accuracy_score(test_Y, y_pred))


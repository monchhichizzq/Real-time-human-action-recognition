# 4 generation: Quantized weight, State Saturation aware training.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM
from keras.layers import Input, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Layer
from keras.layers import Conv1D, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling1D, AveragePooling1D
from keras.optimizers import SGD, Adam
from keras import regularizers



def HG_model(input_shape, num_classes, r = 1e-3) :
    main_input = Input(shape=input_shape, name='main_input')
    lstm1 = LSTM(16, use_bias=True, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(1e-1),
                 bias_initializer='zeros', return_sequences=True)(main_input)
    dropout1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(16, use_bias=True, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(1e-1),
                 bias_initializer='zeros', return_sequences=False)(dropout1)
    dropout2 = Dropout(0.5)(lstm2)
    dense = Dense(num_classes)(dropout2)
    predictions = Activation('softmax')(dense)
    model = Model(inputs=main_input, outputs=predictions)
    model.summary()
    return model


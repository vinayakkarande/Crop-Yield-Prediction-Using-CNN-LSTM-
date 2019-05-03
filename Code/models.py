from keras import regularizers
from keras.optimizers import Nadam, SGD, Adam
from keras.layers import Input, LSTM, Embedding, Dense, LeakyReLU, Flatten, Dropout, SeparableConv2D, GlobalAveragePooling3D
from keras.layers import TimeDistributed, BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv1D, MaxPooling3D, Conv3D, ConvLSTM2D, LSTM, AveragePooling2D
from keras import optimizers
from keras.callbacks import EarlyStopping
import sys

class Model_List():
    def __init__(self, model, frames, dimensions, saved_model=None, print_model=False):
        self.frames = frames
        self.saved_model = saved_model
        self.image_dim = tuple(dimensions)
        self.input_shape = (frames, ) + tuple(dimensions)
        self.print_model = print_model

        metrics = ['mean_absolute_error']

        # Select model.
        if self.saved_model is not None:
            print("Loading model %s" % self.saved_model)
            self.model = load_model(self.saved_model)
        elif model == 'CNN_AVG_reg':
            print("Loading Model.")
            self.model = self.CNN_AVG_reg()
        elif model == 'CNN_MAX_reg':
            print("Loading Model.")
            self.model = self.CNN_MAX_reg()
        elif model == 'CNN_LSTM_AVG':
            print("Loading Model.")
            self.model = self.CNN_LSTM_AVG()
        
        elif model == 'CNN_LSTM':
            print("Loading Model.")
            self.model = self.CNN_LSTM()
            
        elif model == 'SepCNN_LSTM':
            print("Loading Model.")
            self.model = self.SepCNN_LSTM()
        elif model == 'CONVLSTM':
            print("Loading Model.")
            self.model = self.CONVLSTM()
        else:
            print("Model not defined")
            sys.exit()

        # Now compile the network.
        optimizer = Adam()
        self.model.compile(loss='mse', optimizer=optimizer, metrics=metrics)

        if self.print_model == True:
            print(self.model.summary())
                           
                           
    def CNN_AVG_reg(self):
        #frames_input = Input(shape=self.input_shape)
        model = Sequential()
        model.add(Conv2D(32, (3, 3),kernel_regularizer=regularizers.l2(0.001), activation='relu', padding='same', input_shape=(38, 256, 10)))
        model.add(AveragePooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3),kernel_regularizer=regularizers.l2(0.001), activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
        model.add(Dense(1))
        return model
     
    def CNN_MAX_reg(self):
        #frames_input = Input(shape=self.input_shape)
        model = Sequential()
        model.add(Conv2D(32, (3, 3),kernel_regularizer=regularizers.l2(0.001),activation='relu', padding='same', input_shape=(38, 256, 10)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3),kernel_regularizer=regularizers.l2(0.001), activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(64,kernel_regularizer=regularizers.l2(0.001), activation='relu'))
        model.add(Dense(1))
        return model 
    
    def CNN_LSTM_AVG(self):

        input_frames = Input(shape=self.input_shape)
        model = Sequential()
        model.add(Conv2D(32, (3, 3),kernel_regularizer=regularizers.l2(0.001), activation='relu', padding='same', input_shape=self.image_dim))
        model.add(AveragePooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3),kernel_regularizer=regularizers.l2(0.001), activation='relu'))
        model.add(Flatten())
        encoded_frame_sequence = TimeDistributed(model)(frames_input) # the output will be a sequence of vectors
        encoded_video = LSTM(256, activation='tanh', return_sequences=True)(encoded_frame_sequence)  # the output will be a vector
        fc2 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(encoded_video)
        out = Flatten()(fc2)
        out = Dropout(0.5)(out)
        output = Dense(1, activation='relu')(out)
        CNN_LSTM_AVG = Model(inputs=input_frames, outputs=output)
        return CNN_LSTM_AVG
    
    def CNN_LSTM(self):

        input_frames = Input(shape=self.input_shape)
        vision_model = Sequential()
        vision_model.add(Conv2D(64, (1, 2), activation='relu', padding='same', input_shape=self.image_dim))
        vision_model.add(BatchNormalization())
        vision_model.add(MaxPooling2D((1, 2)))
        vision_model.add(Flatten())
        vision_model.add(BatchNormalization())
        encoded_frame_sequence = TimeDistributed(vision_model)(frames_input) # the output will be a sequence of vectors
        encoded_video = LSTM(256, activation='tanh', return_sequences=True)(encoded_frame_sequence)  # the output will be a vector
        
        fc2 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05))(encoded_video)
        out = Flatten()(fc2)
        out = Dropout(0.5)(out)
        output = Dense(1, activation='relu')(out)
        CNN_LSTM = Model(inputs=input_frames, outputs=output)
        return CNN_LSTM
    
    
    def SepCNN_LSTM(self):

        frames_input = Input(shape=self.input_shape)
        
        vision_model = Sequential()
        vision_model.add(SeparableConv2D(64, (1, 2), activation='relu', padding='same', input_shape=self.image_dim))
        vision_model.add(BatchNormalization())
        vision_model.add(MaxPooling2D((1, 2)))
        vision_model.add(Flatten())
        vision_model.add(BatchNormalization())
        encoded_frame_sequence = TimeDistributed(vision_model)(frames_input) # the output will be a sequence of vectors
        encoded_video = LSTM(256, activation='tanh', return_sequences=True)(encoded_frame_sequence)  # the output will be a vector
        
        fc2 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05))(encoded_video)
        out = Flatten()(fc2)
        out = Dropout(0.5)(out)
        output = Dense(1, activation='relu')(out)
        CNN_LSTM = Model(inputs=frames_input, outputs=output)
        
        return CNN_LSTM
    
    
    def CONVLSTM(self):

        CONVLSTM = Sequential()
        CONVLSTM.add(ConvLSTM2D(filters=64, kernel_size=(1, 2),
                   input_shape=self.input_shape,
                   padding='same', return_sequences=True,
                   activation='relu'))
        CONVLSTM.add(ConvLSTM2D(filters=32, kernel_size=(1, 2),
                   padding='same', return_sequences=True,
                   activation='relu'))
        CONVLSTM.add(ConvLSTM2D(filters=32, kernel_size=(1, 2),
                   padding='same', return_sequences=True,
                   activation='relu'))
        CONVLSTM.add(BatchNormalization())
        CONVLSTM.add(Flatten())

        CONVLSTM.add(Dense(32, activation='relu'))
        CONVLSTM.add(Dropout(0.5))
        CONVLSTM.add(Dense(1, activation='relu'))
        
        return CONVLSTM
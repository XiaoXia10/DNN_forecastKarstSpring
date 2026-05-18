# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:56:57 2024

@author: Xiao Xia Liang
"""
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed, GRU
from tensorflow.keras import initializers, callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

    
def auto_encoder_lstm(train_x, train_y, args):
    
    print("LSTM model")
    # Define the encoder
    encoder_inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
    encoder = LSTM(args.latent_dim,
                    dropout=args.dropout, 
                    recurrent_dropout=args.recurrent_dropout, 
                    return_state=True)

    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Define the decoder
    decoder_inputs = RepeatVector(train_y.shape[1])(encoder_outputs)
    decoder_lstm = LSTM(args.latent_dim, 
                        dropout=args.dropout, 
                        recurrent_dropout=args.recurrent_dropout, 
                        return_sequences=True, 
                        return_state=False)

    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = TimeDistributed(Dense(train_y.shape[2], activation='linear'))
    
    decoder_outputs = decoder_dense(decoder_outputs)
    # Define the encoder-decoder model
    model = Model(encoder_inputs, decoder_outputs)
    
    adam_optimizer = Adam(learning_rate=args.learning_rate)
    
    print("MAE Loss function is used")
    model.compile(optimizer=adam_optimizer, loss="mae")
    
    return model


def auto_encoder_gru(train_x, train_y, args):
    print("GRU model")
    # Define the encoder
    encoder_inputs = Input(shape=(train_x.shape[1], train_x.shape[2]))
    encoder = GRU(args.latent_dim,
                    dropout=args.dropout, 
                    recurrent_dropout=args.recurrent_dropout, 
                    return_state=True)

    encoder_outputs, state_h = encoder(encoder_inputs)

    # Define the decoder
    decoder_inputs = RepeatVector(train_y.shape[1])(encoder_outputs)
    decoder_ = GRU(args.latent_dim, 
                        dropout=args.dropout, 
                        recurrent_dropout=args.recurrent_dropout, 
                        return_sequences=True, 
                        return_state=False)

    decoder_outputs = decoder_(decoder_inputs, initial_state=state_h)
    decoder_dense = TimeDistributed(Dense(train_y.shape[2], activation='linear'))
    
    decoder_outputs = decoder_dense(decoder_outputs)
    # Define the encoder-decoder model
    model = Model(encoder_inputs, decoder_outputs)
    
    adam_optimizer = Adam(learning_rate=args.learning_rate)
    
    print("MAE Loss function is used")
    model.compile(optimizer=adam_optimizer, loss="mae")
    
    return model




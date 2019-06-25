from __future__ import print_function

from keras.models import Model, Sequential, load_model
from keras.layers import Embedding, Dense, Input, RepeatVector, TimeDistributed, concatenate, add, Dropout
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

HIDDEN_UNITS = 100
DEFAULT_BATCH_SIZE = 64
VERBOSE = 1
DEFAULT_EPOCHS = 10

class OneShotRNN(object):
    model_name = 'one-shot-rnn'
    """
    The first alternative model is to generate the entire output sequence in a one-shot manner.
    That is, the decoder uses the context vector alone to generate the output sequence.

    This model puts a heavy burden on the decoder.
    It is likely that the decoder will not have sufficient context for generating a coherent output sequence as it
    must choose the words and their order.
    """

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.config = config
        self.version = 0
        if 'version' in config:
            self.version = config['version']

        print('max_input_seq_length', self.max_input_seq_length)
        print('max_target_seq_length', self.max_target_seq_length)
        print('num_input_tokens', self.num_input_tokens)
        print('num_target_tokens', self.num_target_tokens)

        # encoder input model
        model = Sequential()
        model.add(Embedding(output_dim=128, input_dim=self.num_input_tokens, input_length=self.max_input_seq_length))

        # encoder model
        model.add(LSTM(128))
        model.add(RepeatVector(self.max_target_seq_length))
        # decoder model
        model.add(LSTM(128, return_sequences=True))
        model.add(TimeDistributed(Dense(self.num_target_tokens, activation='softmax')))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.input_word2idx:
                    wid = self.input_word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_target_data_batch[lineIdx, idx, w2idx] = 1
                yield encoder_input_data_batch, decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + OneShotRNN.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + OneShotRNN.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + OneShotRNN.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, model_dir_path=None, batch_size=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version

        config_file_path = OneShotRNN.get_config_file_path(model_dir_path)
        weight_file_path = OneShotRNN.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = OneShotRNN.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)
        predicted = self.model.predict(input_seq)
        predicted_word_idx_list = np.argmax(predicted, axis=1)
        predicted_word_list = [self.target_idx2word[wid] for wid in predicted_word_idx_list[0]]
        return predicted_word_list


def main():

    # model_dir_path = './models'
    #
    # config = np.load(OneShotRNN.get_config_file_path(model_dir_path=model_dir_path)).item()
    #
    # model = OneShotRNN(config)
    # model.load_weights(weight_file_path=OneShotRNN.get_weight_file_path(model_dir_path=model_dir_path))

    model = load_model("./models/one-shot-rnn-weights.h5")

    print('model loaded')

    model.save('./model.h5')

    print('model saved')

if __name__ == '__main__':
    main()

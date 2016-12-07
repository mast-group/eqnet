import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from deeplearning.optimization import dropout_multiple


# theano.config.compute_test_value = 'warn'

class AbstractLayer:
    def get_params(self) -> dict:
        """
        Get the trainable parameters of the layer
        """
        raise NotImplementedError()

    @property
    def name(self) -> str:
        raise NotImplementedError()


class RecurrentCell(AbstractLayer):
    def get_cell_with_dropout(self, rng: RandomStreams, dropout_rate: float):
        raise NotImplementedError()

    def get_next_state(self, previous_state, input):
        raise NotImplementedError()


class SimpleRecurrentCell(RecurrentCell):
    def __init__(self, memory_D: int, input_D: int, name: str, log_init_scale: float):
        self.__name = name
        self.__memory_D = memory_D
        self.__input_D = input_D

        prev_hidden_to_next = np.random.randn(memory_D, memory_D) * 10 ** log_init_scale
        self.__prev_hidden_to_next = theano.shared(prev_hidden_to_next.astype(theano.config.floatX),
                                                   name=name + "_rnn_prev_hidden_to_next")

        prediction_to_hidden = np.random.randn(input_D, memory_D) * 10 ** log_init_scale
        self.__prediction_to_hidden = theano.shared(prediction_to_hidden.astype(theano.config.floatX),
                                                    name=name + "_rnn_prediction_to_hidden")

        bias = np.random.randn(memory_D) * 10 ** log_init_scale
        self.__bias = theano.shared(bias.astype(theano.config.floatX), name=name + "_bias")

    def get_params(self) -> dict:
        return {
            self.__name + ":rnn_prev_hidden_to_next": self.__prev_hidden_to_next,
            self.__name + ":rnn_prediction_to_hidden": self.__prediction_to_hidden,
            self.__name + ":rnn_bias": self.__bias
        }

    @property
    def name(self) -> str:
        return self.__name

    def get_cell_with_dropout(self, rng: RandomStreams, dropout_rate: float):
        with_dropout = SimpleRecurrentCell.__new__(self.__class__)

        with_dropout.__prev_hidden_to_next, with_dropout.__prediction_to_hidden = dropout_multiple(
            dropout_rate, rng, True, self.__prev_hidden_to_next, self.__prediction_to_hidden)
        with_dropout.__bias = self.__bias
        with_dropout.get_cell_with_dropout = None
        with_dropout.__name = self.__name + ":with_dropout"
        return with_dropout

    def get_next_state(self, previous_state, input):
        next_hidden = T.tanh(T.dot(previous_state, self.__prev_hidden_to_next)
                             + T.dot(input, self.__prediction_to_hidden) + self.__bias)

        return next_hidden


class LinearRecurrentCell(RecurrentCell):
    def __init__(self, memory_D: int, input_D: int, name: str, log_init_scale: float):
        self.__name = name
        self.__memory_D = memory_D
        self.__input_D = input_D

        prev_hidden_to_next = np.random.randn(memory_D, memory_D) * 10 ** log_init_scale
        self.__prev_hidden_to_next = theano.shared(prev_hidden_to_next.astype(theano.config.floatX),
                                                   name=name + "_rnn_prev_hidden_to_next")

        prediction_to_hidden = np.random.randn(input_D, memory_D) * 10 ** log_init_scale
        self.__prediction_to_hidden = theano.shared(prediction_to_hidden.astype(theano.config.floatX),
                                                    name=name + "_rnn_prediction_to_hidden")

        bias = np.random.randn(memory_D) * 10 ** log_init_scale
        self.__bias = theano.shared(bias.astype(theano.config.floatX), name=name + "_bias")

    def get_params(self) -> dict:
        return {
            self.__name + ":rnn_prev_hidden_to_next": self.__prev_hidden_to_next,
            self.__name + ":rnn_prediction_to_hidden": self.__prediction_to_hidden,
            self.__name + ":rnn_bias": self.__bias
        }

    @property
    def name(self) -> str:
        return self.__name

    def get_cell_with_dropout(self, rng: RandomStreams, dropout_rate: float):
        with_dropout = LinearRecurrentCell.__new__(LinearRecurrentCell)

        with_dropout.__prev_hidden_to_next, with_dropout.__prediction_to_hidden = dropout_multiple(
            dropout_rate, rng, True, self.__prev_hidden_to_next, self.__prediction_to_hidden)
        with_dropout.__bias = self.__bias
        with_dropout.get_cell_with_dropout = None
        with_dropout.__name = self.__name + ":with_dropout"
        return with_dropout

    def get_next_state(self, previous_state, input):
        next_hidden = T.dot(previous_state, self.__prev_hidden_to_next) \
                      + T.dot(input, self.__prediction_to_hidden) + self.__bias

        return next_hidden


class GruCell(RecurrentCell):
    def __init__(self, memory_D: int, input_D: int, name: str, log_init_scale: float, grad_clip: float = None):
        self.__name = name
        self.__memory_D = memory_D
        self.__input_D = input_D
        if grad_clip is not None:
            assert grad_clip > 0
        self.__grad_clip = grad_clip

        w_hid = np.random.randn(memory_D, 3 * memory_D) * 10 ** log_init_scale
        self.__w_hid = theano.shared(w_hid.astype(theano.config.floatX),
                                     name=name + "_w_hid")

        w_in = np.random.randn(input_D, 3 * memory_D) * 10 ** log_init_scale
        self.__w_in = theano.shared(w_in.astype(theano.config.floatX),
                                    name=name + "_w_in")

        biases = np.random.randn(3 * memory_D) * 10 ** log_init_scale
        self.__biases = theano.shared(biases.astype(theano.config.floatX),
                                      name=name + "_biases")

    def get_params(self) -> dict:
        return {
            self.__name + ":w_hid": self.__w_hid,
            self.__name + ":w_in": self.__w_in,
            self.__name + ":biases": self.__biases
        }

    @property
    def name(self) -> str:
        return self.__name

    def get_cell_with_dropout(self, rng: RandomStreams, dropout_rate: float):
        with_dropout = GruCell.__new__(GruCell)

        with_dropout.__w_hid, with_dropout.__w_in = dropout_multiple(
            dropout_rate, rng, True, self.__w_hid, self.__w_in)
        with_dropout.__biases = self.__biases
        with_dropout.get_cell_with_dropout = None
        with_dropout.__name = self.__name + ":with_dropout"
        with_dropout.__memory_D = self.__memory_D
        with_dropout.__grad_clip = self.__grad_clip
        return with_dropout

    def get_next_state(self, previous_state, input):
        w_in, w_hid, biases = self.__w_in, self.__w_hid, self.__biases

        input_D_stacked = T.dot(input, w_in)
        prev_D_stacked = T.dot(previous_state, w_hid)

        gates = T.nnet.sigmoid(
            input_D_stacked[:2 * self.__memory_D] + prev_D_stacked[:2 * self.__memory_D] + self.__biases[
                                                                                           :2 * self.__memory_D])
        reset_gate, update_gate = gates[:self.__memory_D], gates[self.__memory_D:]

        hidden_update = input_D_stacked[2 * self.__memory_D:] + reset_gate * prev_D_stacked[
                                                                             2 * self.__memory_D:] + self.__biases[
                                                                                                     2 * self.__memory_D:]
        if self.__grad_clip is not None:
            hidden_update = theano.gradient.grad_clip(hidden_update, -self.__grad_clip, self.__grad_clip)
        hidden_update = T.tanh(hidden_update)

        next_hidden = (1. - update_gate) * previous_state + update_gate * hidden_update
        return next_hidden


class GRU(AbstractLayer):
    """
    A sequence GRU
    """

    def name(self):
        return self.__name

    def __init__(self, embeddings, memory_size: int, embeddings_size: int, hyperparameters: dict, rng: RandomStreams,
                 name="SequenceGRU", use_centroid=False):
        """
        :param embeddings: the embedding matrix
        """
        self.__name = name
        self.__embeddings = embeddings
        self.__memory_size = memory_size
        self.__embeddings_size = embeddings_size
        self.__hyperparameters = hyperparameters
        self.__rng = rng

        if use_centroid:
            self.__gru = GruCentroidsCell(memory_size, embeddings_size, hyperparameters['num_centroids'],
                                          hyperparameters['centroid_use_rate'], self.__rng, self.__name + ":GRUCell",
                                          hyperparameters['log_init_noise'])
        else:
            self.__gru = GruCell(memory_size, embeddings_size, self.__name + ":GRUCell",
                                 hyperparameters['log_init_noise'])

        self.__params = {self.__name + ":" + n: v for n, v in self.__gru.get_params().items()}

    def __get_hidden_state(self, input_sequence, initial_state, use_dropout):
        gru_cell = self.__gru.get_cell_with_dropout(self.__rng, self.__hyperparameters[
            'dropout_rate']) if use_dropout else self.__gru

        def encode_step(input_token_id, prev_hidden_state, *args):
            input_token_embedding = self.__embeddings[input_token_id]
            return gru_cell.get_next_state(prev_hidden_state, input_token_embedding)

        non_sequences = list(gru_cell.get_params().values()) + [self.__embeddings]
        h, _ = theano.scan(encode_step, sequences=input_sequence, outputs_info=[initial_state],
                           name=self.__name + ":sequence_scan", non_sequences=non_sequences,
                           strict=True)
        return h

    def get_params(self):
        return self.__params

    def get_encoding(self, input_sequence, initial_state, use_dropout=False):
        return self.__get_hidden_state(input_sequence, initial_state, use_dropout)[-1]

    def get_all_hidden_states(self, input_sequence, initial_state, use_dropout=False):
        return self.__get_hidden_state(input_sequence, initial_state, use_dropout)


class AveragingGRU(AbstractLayer):
    """
    A sequence GRU
    """

    def name(self):
        return self.__name

    def __init__(self, embeddings, memory_size: int, embeddings_size: int, hyperparameters: dict, rng: RandomStreams,
                 name="SequenceAveragingGRU", use_centroid=False):
        """
        :param embeddings: the embedding matrix
        """
        self.__name = name
        self.__embeddings = embeddings
        self.__memory_size = memory_size
        self.__embeddings_size = embeddings_size
        self.__hyperparameters = hyperparameters
        self.__rng = rng

        if use_centroid:
            self.__gru = GruCentroidsCell(memory_size, embeddings_size, hyperparameters['num_centroids'],
                                          hyperparameters['centroid_use_rate'], self.__rng, self.__name + ":GRUCell",
                                          hyperparameters['log_init_noise'])
        else:
            self.__gru = GruCell(memory_size, embeddings_size, self.__name + ":GRUCell",
                                 hyperparameters['log_init_noise'])

        self.__params = {self.__name + ":" + n: v for n, v in self.__gru.get_params().items()}

    def __get_hidden_state(self, input_sequence, initial_state, use_dropout):
        gru_cell = self.__gru.get_cell_with_dropout(self.__rng, self.__hyperparameters[
            'dropout_rate']) if use_dropout else self.__gru

        def encode_step(input_token_id, prev_hidden_state, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9):
            input_token_embedding = self.__embeddings[input_token_id]
            return gru_cell.get_next_state(prev_hidden_state, input_token_embedding)

        non_sequences = list(gru_cell.get_params().values()) + [self.__embeddings]
        h, _ = theano.scan(encode_step, sequences=input_sequence, outputs_info=[initial_state],
                           name=self.__name + ":sequence_scan", non_sequences=non_sequences,
                           strict=True)
        return h

    def get_params(self):
        return self.__params

    def get_encoding(self, input_sequence, initial_state, use_dropout=False):
        return T.sum(self.__get_hidden_state(input_sequence, initial_state, use_dropout), axis=0)

    def get_all_hidden_states(self, input_sequence, initial_state, use_dropout=False):
        return self.__get_hidden_state(input_sequence, initial_state, use_dropout)

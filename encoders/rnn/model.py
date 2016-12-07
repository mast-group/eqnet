import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from deeplearning.layers import AbstractLayer
from deeplearning.optimization import dropout, dropout_multiple
from deeplearning.utils import Bunch
from encoders.rnn.treedata import TreeDatasetExtractor


class RNN(AbstractLayer):
    def __init__(self, memory_size: int, hyperparameters: dict, rng: RandomStreams,
                 treedata: TreeDatasetExtractor, name: str = "TreeRnn", combination_type='eqnet'):
        self.__name = name
        self.__memory_size = memory_size
        self.__hyperparameters = hyperparameters
        self.__combination_type = combination_type
        self.__rng = rng
        self.__treedata = treedata

        self.__terminal_idx = T.ivector(name + ":terminal_idx")
        self.__terminal_types = T.ivector(name + ":terminal_types")
        self.__current_idx = T.ivector(name + ":current_idx")
        self.__children_idxs = T.imatrix(name + ":child_idxs")
        self.__node_types = T.ivector(name + ":node_types")
        self.__node_symbols = T.iscalar(name + ":node_symbols")
        self.__num_nodes = T.iscalar(name + ":num_nodes")

        # Used only for the leaf node representations
        node_embeddings = np.random.randn(len(treedata.node_type_dictionary), memory_size) * \
                          10 ** self.__hyperparameters["log_init_scale_embedding"]
        self.__node_embeddings = theano.shared(node_embeddings.astype(theano.config.floatX),
                                               name=name + ":terminal_embeddings")

        self.__dropped_out_params = Bunch()
        self.__dropped_out_params.node_embeddings_with_dropout = self.__node_embeddings + \
                                                                 self.__rng.normal(size=self.__node_embeddings.shape,
                                                                                   std=10 ** self.__hyperparameters[
                                                                                       "log_init_scale_embedding"])

        self.__required_hyperparameters = []

        if combination_type == 'single':
            self.__parent_state_combiner = SingleLayerCombination(self.__memory_size,
                                                                  len(treedata.node_type_dictionary),
                                                                  self.__treedata.max_num_properties_per_node,
                                                                  self.__hyperparameters, self.__rng)
        elif combination_type == 'eqnet':
            hidden_layer_sizes = self.__hyperparameters['hidden_layer_sizes']
            self.__required_hyperparameters.extend(
                ['hidden_layer_sizes', 'ae_representation_size', 'ae_noise', 'constrain_intro_rate'])
            self.__parent_state_combiner = ResidualWithAutoencoder(
                self.__memory_size, len(treedata.node_type_dictionary),
                self.__treedata.max_num_properties_per_node,
                self.__hyperparameters, self.__rng, hidden_layer_sizes)
        elif combination_type == 'double':
            hidden_layer_size = self.__hyperparameters['hidden_layer_size']
            self.__required_hyperparameters.extend(['hidden_layer_size'])
            self.__parent_state_combiner = TwoLayerCombination(self.__memory_size, len(treedata.node_type_dictionary),
                                                               self.__treedata.max_num_properties_per_node,
                                                               self.__hyperparameters, hidden_layer_size, self.__rng)
        else:
            raise Exception("Unrecognized state combinator '" + combination_type + "'")

    def name(self):
        return self.__name

    @property
    def required_hyperparameters(self) -> set:
        return set(self.__required_hyperparameters)

    def get_input_variables(self):
        return TreeDatasetExtractor.TreeArrayRepresentation(self.__terminal_idx, self.__terminal_types,
                                                            self.__current_idx, self.__children_idxs,
                                                            self.__node_types, self.__num_nodes, self.__node_symbols)

    def get_params(self):
        params = self.__parent_state_combiner.get_params()
        assert 'leaf_embeddings' not in params
        params['leaf_embeddings'] = self.__node_embeddings
        return params

    def copy_full(self):
        copy = RNN(self.__memory_size, self.__hyperparameters, self.__rng, self.__treedata, name=self.__name + ":copy",
                   combination_type=self.__combination_type)
        copy.get_params = dict
        copy.__node_embeddings = self.__node_embeddings
        copy.__parent_state_combiner = self.__parent_state_combiner
        copy.__dropped_out_params = self.__dropped_out_params
        return copy

    def get_encoding(self, use_dropout, iteration_number=0) -> tuple:
        combiner = self.__parent_state_combiner
        combiner_params = combiner.get_droppedout_params(forscan=True) if use_dropout else combiner.get_params(
            forscan=True)

        embeddings = self.__dropped_out_params.node_embeddings_with_dropout if use_dropout else self.__node_embeddings
        leaf_embeddings, leaf_additional_objective = combiner.transform_leaf_embeddings(
            embeddings[self.__terminal_types], use_dropout)
        # By conventions, the last one is 0s to be used as a placeholder
        initial_node_states = T.zeros((self.__num_nodes + 1, self.__memory_size), dtype=theano.config.floatX)
        initial_node_states = T.set_subtensor(initial_node_states[self.__terminal_idx], leaf_embeddings)

        non_sequences = list(combiner_params.values())
        if iteration_number != 0:
            non_sequences += [iteration_number]

        def get_next_state(current_idx, child_idxs, node_type, previous_state, previous_additional_objective, *args):
            node_state, add_to_objective = combiner.get_parent_state(previous_state[child_idxs], node_type, use_dropout,
                                                                     iteration_number)
            return T.set_subtensor(previous_state[current_idx],
                                   node_state), previous_additional_objective + add_to_objective

        [node_states, additional_objective], _ = theano.scan(fn=get_next_state,
                                                             sequences=[self.__current_idx, self.__children_idxs,
                                                                        self.__node_types],
                                                             outputs_info=[initial_node_states,
                                                                           leaf_additional_objective],
                                                             name=self.__name + ":sequence_scan",
                                                             non_sequences=non_sequences, strict=True)
        return node_states[-1][-2], node_states[-1, :-1], additional_objective[-1] / (
        self.__num_nodes + 1)  # The last one is always 0s, pick the second last


class ChildrenToParentStateCombiner(AbstractLayer):
    def get_parent_state(self, children_states, node_type, use_dropout: bool, iteration_number) -> tuple:
        """
        Return the parent state of the recurrent NN and any additional terms to be added to the objective (objective is maximized)
        """
        raise NotImplemented("Abstract Class")

    def transform_leaf_embeddings(self, leaf_embeddings, use_dropout: bool):
        """
        Make any necessary transformations to leaf embeddings. By default none are made.
        """
        return leaf_embeddings, 0

    def get_droppedout_params(self, forscan: bool = False):
        raise NotImplemented("Abstract Class")

    def get_params(self, forscan: bool = False):
        raise NotImplemented("Abstract Class")


class SingleLayerCombination(ChildrenToParentStateCombiner):
    def __init__(self, memory_size: int, num_node_types: int, max_num_children: int, hyperparameters: dict,
                 rng: RandomStreams, name: str = "single_layer_combination"):
        self.__memory_size = memory_size
        self.__rng = rng
        self.__name = name
        self.__hyperparameters = hyperparameters

        w = np.random.randn(num_node_types, memory_size, max_num_children * memory_size) * \
            10 ** self.__hyperparameters["log_init_scale_embedding"]
        self.__w = theano.shared(w.astype(theano.config.floatX), name=name + ":w")

        bias = np.random.randn(num_node_types, memory_size) * 10 ** self.__hyperparameters["log_init_scale_embedding"]
        self.__bias = theano.shared(bias.astype(theano.config.floatX), name=name + ":b")

        self.__w_with_dropout = \
            dropout(self.__hyperparameters['dropout_rate'], self.__rng, self.__w, True)

    def name(self):
        return self.__name

    def get_params(self, forscan: bool = False):
        return dict(w=self.__w, bias=self.__bias)

    def get_droppedout_params(self, forscan: bool = False):
        params = dict(node_embeddings=self.__w_with_dropout, bias=self.__bias)
        return params

    def get_parent_state(self, children_states, node_type, use_dropout: bool, iteration_number) -> tuple:
        w = self.__w_with_dropout if use_dropout else self.__w
        return T.tanh(T.dot(w[node_type], T.flatten(children_states)) + self.__bias[node_type]), 0


class TwoLayerCombination(ChildrenToParentStateCombiner):
    def __init__(self, memory_size: int, num_node_types: int, max_num_children: int, hyperparameters: dict,
                 hidden_layer_size: int, rng: RandomStreams, name: str = "single_layer_combination"):
        self.__memory_size = memory_size
        self.__rng = rng
        self.__name = name
        self.__hyperparameters = hyperparameters

        w_l1 = np.random.randn(num_node_types, hidden_layer_size, max_num_children * memory_size) * \
               10 ** self.__hyperparameters["log_init_scale_embedding"]
        self.__w_l1 = theano.shared(w_l1.astype(theano.config.floatX), name=name + ":w_l1")

        bias_l1 = np.random.randn(num_node_types, hidden_layer_size) * 10 ** self.__hyperparameters[
            "log_init_scale_embedding"]
        self.__bias_l1 = theano.shared(bias_l1.astype(theano.config.floatX), name=name + ":b_l1")

        w_l2 = np.random.randn(num_node_types, memory_size, hidden_layer_size) * \
               10 ** self.__hyperparameters["log_init_scale_embedding"]
        self.__w_l2 = theano.shared(w_l2.astype(theano.config.floatX), name=name + ":w_l2")

        bias_l2 = np.random.randn(num_node_types, memory_size) * 10 ** self.__hyperparameters[
            "log_init_scale_embedding"]
        self.__bias_l2 = theano.shared(bias_l2.astype(theano.config.floatX), name=name + ":b_l2")

        self.__w_l1_with_dropout, self.__w_l2_with_dropout = \
            dropout_multiple(self.__hyperparameters['dropout_rate'], self.__rng, True, self.__w_l1, self.__w_l2)

    def name(self):
        return self.__name

    def get_params(self, forscan: bool = False):
        return dict(w_l1=self.__w_l1, bias_l1=self.__bias_l1, w_l2=self.__w_l2, bias_l2=self.__bias_l2)

    def get_droppedout_params(self, forscan: bool = False):
        params = dict(w_l1=self.__w_l1_with_dropout, bias_l1=self.__bias_l1,
                      w_l2=self.__w_l2_with_dropout, bias_l2=self.__bias_l2)
        return params

    def get_parent_state(self, children_states, node_type, use_dropout: bool, iteration_number) -> tuple:
        w_l1 = self.__w_l1_with_dropout if use_dropout else self.__w_l1
        l1_out = T.tanh(T.dot(w_l1[node_type], T.flatten(children_states)) + self.__bias_l1[node_type])

        w_l2 = self.__w_l2_with_dropout if use_dropout else self.__w_l2
        l2_out = T.tanh(T.dot(w_l2[node_type], l1_out) + self.__bias_l2[node_type])
        return l2_out, 0


class ResidualWithAutoencoder(ChildrenToParentStateCombiner):
    def __init__(self, memory_size: int, num_node_types: int, max_num_children: int, hyperparameters: dict,
                 rng: RandomStreams, hidden_layer_sizes: list, name: str = "residual_with_ae"):
        self.__memory_size = memory_size
        self.__rng = rng
        self.__name = name
        self.__hyperparameters = hyperparameters

        self.__max_num_children = max_num_children
        prev_layer_size = max_num_children * memory_size
        self.__w = []
        self.__bias = []
        self.__params = {}

        # Build n-layer NN
        for i, layer_size in enumerate(hidden_layer_sizes):
            w = np.random.randn(num_node_types, layer_size, prev_layer_size) * \
                10 ** self.__hyperparameters["log_init_scale_embedding"]
            w_shared = theano.shared(w.astype(theano.config.floatX), name=name + ":w_l" + str(i))
            self.__w.append(w_shared)
            self.__params[name + ":w_l" + str(i)] = w_shared

            bias = np.random.randn(num_node_types, layer_size) * 10 ** self.__hyperparameters[
                "log_init_scale_embedding"]
            bias_shared = theano.shared(bias.astype(theano.config.floatX), name=name + ":b_l" + str(i))
            self.__bias.append(bias_shared)
            self.__params[name + ":bias_l" + str(i)] = bias_shared

            prev_layer_size = layer_size

        self.__w_res = []
        self.__bias_res = []
        for i, input_layer_size in enumerate([max_num_children * memory_size] + hidden_layer_sizes):
            w = np.random.randn(num_node_types, memory_size, input_layer_size) * \
                10 ** self.__hyperparameters["log_init_scale_embedding"]
            self.__w_res.append(
                theano.shared(w.astype(theano.config.floatX), name=name + ":w_res_l" + str(len(self.__w_res))))
            self.__params[name + ":w_res_l" + str(len(self.__w_res) - 1)] = self.__w_res[-1]

            bias = np.random.randn(num_node_types, memory_size) * 10 ** self.__hyperparameters[
                "log_init_scale_embedding"]
            self.__bias_res.append(
                theano.shared(bias.astype(theano.config.floatX), name=name + ":b_res_l" + str(len(self.__bias_res))))
            self.__params[name + ":bias_res_l" + str(len(self.__bias_res) - 1)] = self.__bias_res[-1]

        self.__w_dropped_out = [dropout(self.__hyperparameters['dropout_rate'], self.__rng, w, True) for w in self.__w]
        self.__w_res_dropped_out = [dropout(self.__hyperparameters['dropout_rate'], self.__rng, w, True) for w in
                                    self.__w_res]

        encoder_size = self.__hyperparameters['ae_representation_size']
        encoder_weights = np.random.randn(num_node_types, (max_num_children + 1) * memory_size, encoder_size) * 10 ** \
                                                                                                                self.__hyperparameters[
                                                                                                                    "log_init_scale_embedding"]
        self.__encoder_weights = theano.shared(encoder_weights.astype(theano.config.floatX),
                                               name=name + ":encoder_weights")
        self.__params[name + ":encoder_weights"] = self.__encoder_weights

        decoder_weights = np.random.randn(encoder_size, max_num_children * memory_size) * 10 ** self.__hyperparameters[
            "log_init_scale_embedding"]
        self.__decoder_weights = theano.shared(decoder_weights.astype(theano.config.floatX),
                                               name=name + ":decoder_weights")
        self.__params[name + ":decoder_weights"] = self.__decoder_weights

        self.__ae_noise = self.__rng.binomial(size=[(max_num_children + 1) * memory_size],
                                              p=1. - self.__hyperparameters['ae_noise'], dtype=theano.config.floatX)

    def name(self):
        return self.__name

    def get_params(self, forscan: bool = False):
        params = dict(self.__params)
        return params

    def get_droppedout_params(self, forscan: bool = False):
        params = dict(self.__params)
        for i, dropped_param in enumerate(self.__w_dropped_out):
            params[self.__name + ":w_l" + str(i)] = dropped_param

        for i, dropped_param in enumerate(self.__w_res_dropped_out):
            params[self.__name + ":w_res_l" + str(i)] = dropped_param

        if forscan:
            params[self.__name + ":ae_noise"] = self.__ae_noise

        return params

    def transform_leaf_embeddings(self, leaf_embeddings, use_dropout: bool):
        transformed_leaf_embeddings, _ = self.__transform_nn_out(leaf_embeddings, use_dropout)
        return transformed_leaf_embeddings, 0.

    def __transform_nn_out(self, inputs, use_dropout: bool):
        norm = inputs.norm(2, axis=1).dimshuffle(0, 'x')
        inputs = inputs / norm
        return inputs, 0

    def get_parent_state(self, children_states, node_type, use_dropout: bool, iteration_number) -> tuple:
        layer_input = T.flatten(children_states)
        nn_out = self.__compute_layer_output(layer_input, node_type, use_dropout, iteration_number)

        encoder_input = T.flatten(T.concatenate((children_states, nn_out))) * self.__ae_noise
        encoding = T.tanh(T.dot(encoder_input, self.__encoder_weights[node_type]))
        decoded = T.tanh(T.dot(encoding, self.__decoder_weights))
        decoded /= decoded.norm(2) / layer_input.norm(2)

        output_reconstruction = self.__compute_layer_output(decoded, node_type, use_dropout, iteration_number)
        reconstruction_cos = T.dot(nn_out[0], output_reconstruction[0])

        children_reconstruction_cos = T.dot(decoded, layer_input)
        additional_objective = reconstruction_cos + children_reconstruction_cos

        constrain_usage_pct = T.cast(1. - T.pow(self.__hyperparameters['constrain_intro_rate'], iteration_number),
                                     theano.config.floatX)
        return nn_out[0], constrain_usage_pct * additional_objective

    def __compute_layer_output(self, layer_input, node_type, use_dropout, iteration_number):
        weights = self.__w_dropped_out if use_dropout else self.__w
        layers = [layer_input]
        for i, w, b in zip(range(len(weights)), weights, self.__bias):
            layers.append(T.nnet.sigmoid(T.dot(w[node_type], layers[-1]) + b[node_type]))
        res_weights = self.__w_res_dropped_out if use_dropout else self.__w_res
        nn_out = 0
        for input_layer, w, b in zip(layers, res_weights, self.__bias_res):
            nn_out += T.dot(w[node_type], input_layer) + b[node_type]
        transformed_nnout, _ = self.__transform_nn_out(nn_out.dimshuffle('x', 0), use_dropout)
        return transformed_nnout

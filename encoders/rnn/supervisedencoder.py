from collections import defaultdict

import numpy as np
import theano

from data.dataimport import import_data
from encoders.rnn.model import RNN

theano.config.floatX = "float32"

import theano.tensor as T
import time
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import trange

from encoders.baseencoder import AbstractEncoder
from encoders.rnn.treedata import TreeDatasetExtractor
from deeplearning.optimization import nesterov_rmsprop_multiple, dropout, log_softmax
from deeplearning.utils import check_hyperparameters, Bunch


class RecursiveNNSupervisedEncoder(AbstractEncoder):
    def __init__(self, training_filename: str, hyperparameters: dict, combination_type='eqnet'):
        self.__hyperparameters = hyperparameters

        self.__dataset_extractor = TreeDatasetExtractor(training_filename)
        self.__rng = RandomStreams()

        self.__rnn = RNN(self.__hyperparameters['memory_size'], self.__hyperparameters, self.__rng,
                         self.__dataset_extractor, combination_type=combination_type)
        check_hyperparameters(self.REQUIRED_HYPERPARAMETERS | self.__rnn.required_hyperparameters,
                              self.__hyperparameters)

        target_embeddings = np.random.randn(self.__hyperparameters['memory_size'],
                                            self.__dataset_extractor.num_equivalent_classes) * 10 ** \
                                                                                               self.__hyperparameters[
                                                                                                   "log_init_scale_embedding"]
        self.__target_embeddings = theano.shared(target_embeddings.astype(theano.config.floatX),
                                                 name="target_embeddings")
        self.__target_embeddings_dropout = dropout(self.__hyperparameters['dropout_rate'], self.__rng,
                                                   self.__target_embeddings, True)

        self.__target_bias = np.log(self.__dataset_extractor.training_empirical_distribution)

        self.__trainable_params = list(self.__rnn.get_params().values()) + [self.__target_embeddings]

        self.__compiled_methods = None
        self.__trained_parameters = None

    REQUIRED_HYPERPARAMETERS = {'log_learning_rate', 'rmsprop_rho', 'momentum', 'minibatch_size', 'grad_clip',
                                'memory_size', 'log_init_scale_embedding', 'dropout_rate', 'curriculum_initial_size',
                                'curriculum_step', 'accuracy_margin'}

    @property
    def rnn(self):
        return self.__rnn

    @property
    def rng(self):
        return self.__rng

    @property
    def hyperparameters(self):
        return self.__hyperparameters

    @property
    def dataset_extractor(self):
        return self.__dataset_extractor

    @property
    def trained_parameters(self):
        params = {}
        param_names = list(self.__rnn.get_params()) + ["target_embeddings"]
        for param, value in zip(param_names, self.__trained_parameters):
            params[param] = value
        return params

    def __get_loss(self, use_dropout: bool, iteration_number=0):
        _, all_node_encodings, additional_objective = self.__rnn.get_encoding(use_dropout, iteration_number)
        target_embeddings = self.__target_embeddings_dropout if use_dropout else self.__target_embeddings

        s = T.dot(all_node_encodings, target_embeddings) + self.__target_bias
        logprobs = log_softmax(s)

        eq_symbol = self.__rnn.get_input_variables().eq_symbol
        targets = T.extra_ops.to_one_hot(eq_symbol.dimshuffle('x'), self.__dataset_extractor.num_equivalent_classes)
        correct = logprobs[-1, eq_symbol]
        rest = T.max(T.flatten(logprobs[-1, (1 - targets).nonzero()]))
        ll = -T.nnet.relu(rest - correct + self.__hyperparameters['accuracy_margin'])
        return logprobs[-1], ll + additional_objective

    def __compile_train_functions(self):
        iteration_number = T.iscalar(name="iteration_number")
        _, ll = self.__get_loss(True, iteration_number)

        grad = T.grad(ll, self.__trainable_params, add_names=True)

        grad_acc = [theano.shared(np.zeros(param.get_value().shape).astype(theano.config.floatX)) for param in
                    self.__trainable_params] + [theano.shared(0, name="sample_count")]
        inputs = list(self.__rnn.get_input_variables()) + [iteration_number]
        self.__compiled_methods.grad_accumulate = theano.function(
            inputs=inputs,
            updates=[(v, v + g) for v, g in zip(grad_acc, grad)] + [(grad_acc[-1], grad_acc[-1] + 1)],
            outputs=T.mean(ll))

        normalized_grads = [T.switch(grad_acc[-1] > 0, g / grad_acc[-1].astype(theano.config.floatX), g) for g in
                            grad_acc[:-1]]

        step_updates, ratios = nesterov_rmsprop_multiple(self.__trainable_params, normalized_grads,
                                                         learning_rate=10 ** self.__hyperparameters[
                                                             "log_learning_rate"],
                                                         rho=self.__hyperparameters["rmsprop_rho"],
                                                         momentum=self.__hyperparameters["momentum"],
                                                         grad_clip=self.__hyperparameters["grad_clip"],
                                                         output_ratios=True)
        step_updates.extend(
            [(v, T.zeros(v.shape).astype(theano.config.floatX)) for v in grad_acc[:-1]])  # Set accumulators to 0
        step_updates.append((grad_acc[-1], 0))
        self.__compiled_methods.grad_step = theano.function(inputs=[], updates=step_updates, outputs=ratios)

    def __compile_test_functions(self):
        logprobs, ll = self.__get_loss(False)
        inputs = list(self.__rnn.get_input_variables())
        self.__compiled_methods.ll_and_logprobs = theano.function(
            inputs=inputs,
            outputs=[T.mean(ll), logprobs])

        self.__compiled_methods.encode = theano.function(inputs=self.__rnn.get_input_variables()[:-1],
                                                         outputs=self.__rnn.get_encoding(False)[0])

    def __compile_if_needed(self):
        if self.__compiled_methods is None:
            print("Compiling Methods...")
            if self.__trained_parameters is not None:
                self.set_parameter_values(self.__trained_parameters)
            self.__compiled_methods = Bunch()
            self.__compile_test_functions()
            self.__compile_train_functions()
            print("Compilation Finished...")

    def set_parameter_values(self, parameter_values: list):
        for param, value in zip(self.__trainable_params, parameter_values):
            param.set_value(value)

    def save(self, filename: str):
        tmp, self.__compiled_methods = self.__compiled_methods, None
        AbstractEncoder.save(self, filename)
        self.__compiled_methods = tmp

    def get_representation_vector_size(self) -> int:
        return self.__hyperparameters['memory_size']

    def get_encoding(self, data: tuple) -> np.array:
        self.__compile_if_needed()
        converted_tree = self.__dataset_extractor.convert_tree_to_array(data[1], ignore_eq_symbols=True)[:-1]

        return self.__compiled_methods.encode(*converted_tree)

    def prediction_accuracy(self, dataset_file):
        self.__compile_if_needed()
        data = import_data(dataset_file)
        dataset = list((self.__dataset_extractor.get_dataset_for_encoder(data, return_num_tokens=True)))

        correct = 0
        for tree in dataset:
            all_args = list(tree[0])
            ll, logprobs = self.__compiled_methods.ll_and_logprobs(*all_args)
            if np.argmax(logprobs) == all_args[-1]:
                correct += 1
        return correct / len(dataset)

    def train(self, training_file, validation_file, max_iter=5000, patience=50, validation_check_limit=2,
              additional_code_to_run=None) -> tuple:
        self.__compile_if_needed()

        minibatch_size = self.__hyperparameters["minibatch_size"]
        training_data = import_data(training_file)
        training_set = list(self.__dataset_extractor.get_dataset_for_encoder(training_data, return_num_tokens=True))
        validation_set = list(self.__dataset_extractor.get_dataset_for_encoder(import_data(validation_file),
                                                                               return_num_tokens=True))

        print("Num classes: %s" % self.__dataset_extractor.num_equivalent_classes)

        def compute_validation_score() -> float:
            print("Train Accuracy %s" % compute_score(training_set, False, True)[1])
            return compute_score(validation_set)

        def compute_score(dataset, print_score=True, return_accuracy=False) -> float:
            # Get all encodings
            sum_ll = 0.
            correct = 0
            for tree in dataset:
                all_args = list(tree[0])
                ll, logprobs = self.__compiled_methods.ll_and_logprobs(*all_args)
                sum_ll += ll
                if np.argmax(logprobs) == all_args[-1]:
                    correct += 1
            if print_score:
                print("Accuracy: %s, LL: %s" % (correct / len(dataset) * 100, sum_ll / len(dataset)))

            if return_accuracy:
                return sum_ll / len(dataset), (correct / len(dataset) * 100)
            return (correct / len(dataset) * 100)

        if self.__trained_parameters is None:
            best_score = float('-inf')
        else:
            best_score = compute_validation_score()
            print("Previous best validation score: %s" % best_score)

        try:
            print("[%s] Training Started..." % time.asctime())
            ratios = np.zeros(len(self.__trainable_params))
            epochs_not_improved = 0
            historic_data = defaultdict(list)
            # Clump minibatches and disallow minibatches that are smaller than their given size, since they may
            # cause instability.
            current_max_size = self.__hyperparameters['curriculum_initial_size']
            curriculum_step = self.__hyperparameters['curriculum_step']
            for i in range(max_iter):
                sample_ordering = []
                for j, tree_data in enumerate(training_set):
                    if tree_data[1] <= current_max_size:
                        sample_ordering.append(j)
                current_max_size += curriculum_step
                np.random.shuffle(np.array(sample_ordering, dtype=np.int32))
                n_batches = 0
                sum_train_loss = 0
                num_elements = 0

                num_minibatches = max(1, min(int(np.floor(float(len(sample_ordering)) / minibatch_size)), 10))

                for j in trange(num_minibatches, desc="Minibatch"):
                    for k in trange(j * minibatch_size, min((j + 1) * minibatch_size, len(sample_ordering)),
                                    desc="Sample", leave=False):
                        current_idx = sample_ordering[k]
                        args = list(training_set[current_idx][0]) + [i]
                        loss = self.__compiled_methods.grad_accumulate(*args)
                        sum_train_loss += loss
                        num_elements += 1

                    n_batches += 1
                    ratios += self.__compiled_methods.grad_step()

                if i % validation_check_limit == validation_check_limit - 1:
                    print("Iteration %s Stats" % i)
                    current_score = compute_validation_score()
                    historic_data['validation_score'].append(current_score)
                    if current_score > best_score:
                        best_score = current_score
                        self.__trained_parameters = [p.get_value() for p in self.__trainable_params]
                        print("At %s validation: current_score=%s [best so far]" % (i, current_score))
                        epochs_not_improved = 0
                    else:
                        print("At %s validation: current_score=%s" % (i, current_score))
                        epochs_not_improved += 1
                    for k in range(len(self.__trainable_params)):
                        print("%s: %.0e" % (self.__trainable_params[k].name, ratios[k] / n_batches))

                    print("Train ll: %s" % (sum_train_loss / num_elements))

                    ratios = np.zeros_like(ratios)
                    if additional_code_to_run is not None: additional_code_to_run(historic_data)
                if epochs_not_improved >= patience:
                    print("Not improved for %s epochs. Stopping..." % patience)
                    break
            print("[%s] Training Finished..." % time.asctime())
        except (InterruptedError, KeyboardInterrupt):
            print("Interrupted. Exiting training gracefully...")

        return best_score, historic_data

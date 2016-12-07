from collections import defaultdict
from itertools import permutations

import numpy as np
import theano
from scipy.spatial.distance import pdist

from data.dataimport import import_data
from encoders.rnn.model import RNN

theano.config.floatX = "float32"

import theano.tensor as T
import time
from theano.tensor.shared_randomstreams import RandomStreams
from tqdm import trange

from encoders.baseencoder import AbstractEncoder
from encoders.rnn.treedata import TreeDatasetExtractor
from deeplearning.optimization import nesterov_rmsprop_multiple
from deeplearning.utils import check_hyperparameters, Bunch


class RecursiveNNSiameseEncoder(AbstractEncoder):
    def __init__(self, training_filename: str, hyperparameters: dict, combination_type='residual_with_ae'):
        self.__hyperparameters = hyperparameters

        self.__dataset_extractor = TreeDatasetExtractor(training_filename)
        self.__rng = RandomStreams()

        self.__rnn = RNN(self.__hyperparameters['memory_size'], self.__hyperparameters, self.__rng,
                         self.__dataset_extractor, combination_type=combination_type)
        self.__trainable_params = list(self.__rnn.get_params().values())
        check_hyperparameters(self.REQUIRED_HYPERPARAMETERS | self.__rnn.required_hyperparameters,
                              self.__hyperparameters)

        self.__compiled_methods = None
        self.__trained_parameters = None

    @staticmethod
    def get_encoder_from_supervised(supervised_encoder, dissimilar_margin: float):
        siamese = RecursiveNNSiameseEncoder.__new__(RecursiveNNSiameseEncoder)
        siamese.__rng = supervised_encoder.rng
        siamese.__rnn = supervised_encoder.rnn
        siamese.__dataset_extractor = supervised_encoder.dataset_extractor
        siamese.__hyperparameters = supervised_encoder.hyperparameters
        siamese.__hyperparameters['dissimilar_margin'] = dissimilar_margin

        siamese.__trainable_params = list(siamese.__rnn.get_params().values())
        saved_parameters = supervised_encoder.trained_parameters
        # print(saved_parameters)
        # siamese.set_parameter_values([saved_parameters[name] for name in siamese.__rnn.get_params()]) # Ignore the target embeddings
        siamese.__trained_parameters = [p.get_value() for p in siamese.__trainable_params]
        siamese.__compiled_methods = None
        return siamese

    REQUIRED_HYPERPARAMETERS = {'log_learning_rate', 'rmsprop_rho', 'momentum', 'minibatch_size', 'grad_clip',
                                'memory_size', 'log_init_scale_embedding', 'dropout_rate', 'dissimilar_margin',
                                'curriculum_initial_size', 'curriculum_step', 'max_num_similar_examples',
                                'max_num_dissimilar_examples'}

    def __get_loss(self, use_dropout, iteration_number=0):
        node_encoding1, _, extra_loss1 = self.__rnn.get_encoding(use_dropout, iteration_number)
        node_encoding1 /= node_encoding1.norm(2)

        copy_rnn = self.__rnn.copy_full()
        node_encoding2, _, extra_loss2 = copy_rnn.get_encoding(use_dropout, iteration_number)
        node_encoding2 /= node_encoding2.norm(2)

        distance = (node_encoding1 - node_encoding2).norm(2)

        are_non_equivalent = self.__rnn.get_input_variables().eq_symbol - copy_rnn.get_input_variables().eq_symbol

        margin = self.__hyperparameters['dissimilar_margin']
        siamese_loss = -T.power(T.switch(are_non_equivalent, T.nnet.relu(margin - distance), distance), 2)
        return siamese_loss + extra_loss1 + extra_loss2, copy_rnn

    def __compile_train_functions(self):
        iteration_number = T.iscalar('iteration_number')
        prob_correct, other_rnn = self.__get_loss(True, iteration_number)

        grad = T.grad(prob_correct, self.__trainable_params, add_names=True)

        grad_acc = [theano.shared(np.zeros(param.get_value().shape).astype(theano.config.floatX)) for param in
                    self.__trainable_params] + [theano.shared(0, name="sample_count")]
        inputs = list(self.__rnn.get_input_variables()) + list(other_rnn.get_input_variables()) + [iteration_number]
        self.__compiled_methods.grad_accumulate = theano.function(
            inputs=inputs,
            updates=[(v, v + g) for v, g in zip(grad_acc, grad)] + [(grad_acc[-1], grad_acc[-1])],
            # TODO: Remove accumulator if indeed not needed
            outputs=T.mean(prob_correct))

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
        prob_correct, other_rnn = self.__get_loss(False)
        inputs = list(self.__rnn.get_input_variables()) + list(other_rnn.get_input_variables())
        self.__compiled_methods.probability = theano.function(
            inputs=inputs,
            outputs=[prob_correct])

        encoding, _, _ = self.__rnn.get_encoding(False)
        encoding /= encoding.norm(2)
        self.__compiled_methods.encode = theano.function(inputs=self.__rnn.get_input_variables()[:-1],
                                                         outputs=encoding)

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
        converted_tree = self.__dataset_extractor.convert_tree_to_array(data[1])[:-1]
        return self.__compiled_methods.encode(*converted_tree)

    def train(self, training_file, validation_file, max_iter=1000, patience=25, validation_check_limit=1,
              additional_code_to_run=None) -> tuple:
        self.__compile_if_needed()

        minibatch_size = self.__hyperparameters["minibatch_size"]
        training_data = import_data(training_file)
        training_set = list(self.__dataset_extractor.get_dataset_for_encoder(training_data, return_num_tokens=True))
        validation_set = list(self.__dataset_extractor.get_dataset_for_encoder(import_data(validation_file),
                                                                               return_num_tokens=True))

        def compute_validation_score() -> float:
            return compute_score(validation_set)

        def compute_score(dataset) -> float:
            # Get all encodings
            encodings = []
            equivalents = defaultdict(set)
            for i, tree in enumerate(dataset):
                encodings.append(self.__compiled_methods.encode(*tree[0][:-1]))
                equivalents[tree[2]].add(i)

            encodings = np.array(encodings, dtype=theano.config.floatX)

            # Get all cosine similarities
            distances = pdist(encodings)

            is_similar = np.zeros_like(distances, dtype=np.int)
            for equivalence_set in equivalents.values():
                for i, j in permutations(equivalence_set, 2):
                    if i > j:
                        is_similar[encodings.shape[0] * j - int(j * (j + 1) / 2) + i - 1 - j] = 1

            similar_score = -np.sum(np.power(distances * is_similar, 2))
            margin = self.__hyperparameters['dissimilar_margin']
            differences = margin - distances
            rectified_diffs = differences * (differences > 0)
            dissimilar_score = -np.sum(np.power(rectified_diffs * (1 - is_similar), 2))

            print("Similar Loss: %s  Dissimilar Loss: %s" % (similar_score, dissimilar_score))
            return similar_score + dissimilar_score

        if self.__trained_parameters is None:
            best_score = float('-inf')
        else:
            best_score = compute_validation_score()
            print("Previous best validation score: %s" % best_score)

        try:
            print("[%s] Training Started..." % time.asctime())
            sum_similar_loss = 0.
            num_similar_loss = 0
            sum_dissimilar_loss = 0.
            num_dissimilar_loss = 0
            ratios = np.zeros(len(self.__trainable_params))
            epochs_not_improved = 0
            historic_data = defaultdict(list)
            # Clump minibatches and disallow minibatches that are smaller than their given size, since they may
            # cause instability.
            num_minibatches = max(1, min(int(np.floor(float(len(training_set)) / minibatch_size)), 10))
            current_max_size = self.__hyperparameters['curriculum_initial_size']
            curriculum_step = self.__hyperparameters['curriculum_step']

            num_examples = self.__hyperparameters['max_num_similar_examples']
            num_dissimilar_examples = self.__hyperparameters['max_num_dissimilar_examples']

            for i in range(max_iter):
                sample_ordering = []
                for j, tree_data in enumerate(training_set):
                    if tree_data[1] <= current_max_size:
                        sample_ordering.append(j)
                current_max_size += curriculum_step
                np.random.shuffle(np.array(sample_ordering, dtype=np.int32))
                n_batches = 0

                for j in trange(num_minibatches, desc="Minibatch"):
                    for k in trange(j * minibatch_size, min((j + 1) * minibatch_size, len(sample_ordering)),
                                    desc="Sample", leave=False):
                        current_idx = sample_ordering[k]
                        # Add siamese gradients, by picking num_examples
                        similar_snippet_idxs = []
                        dissimilar_snippet_idxs = []
                        for l in range(len(sample_ordering)):
                            if l == k:
                                continue
                            other_idx = sample_ordering[l]
                            if training_set[current_idx][2] == training_set[other_idx][2]:
                                similar_snippet_idxs.append(other_idx)
                            else:
                                dissimilar_snippet_idxs.append(other_idx)
                        dissimilar_snippet_idxs = np.array(dissimilar_snippet_idxs)

                        np.random.shuffle(similar_snippet_idxs)
                        np.random.shuffle(dissimilar_snippet_idxs)
                        for other_idx in similar_snippet_idxs[:num_examples]:
                            args = list(training_set[current_idx][0]) + list(training_set[other_idx][0]) + [i]
                            loss = self.__compiled_methods.grad_accumulate(*args)
                            sum_similar_loss += loss
                            num_similar_loss += 1

                        for other_idx in dissimilar_snippet_idxs[:num_dissimilar_examples]:
                            args = list(training_set[current_idx][0]) + list(training_set[other_idx][0]) + [i]
                            loss = self.__compiled_methods.grad_accumulate(*args)
                            sum_dissimilar_loss += loss
                            num_dissimilar_loss += 1 if loss < 0 else 0

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

                    print("Train sum similar-loss: %s (%s samples)" % (sum_similar_loss, num_similar_loss))
                    print("Train sum dissimilar-loss: %s (%s samples)" % (sum_dissimilar_loss, num_dissimilar_loss))
                    # print("Training Set stats: %s" % compute_score(training_set[:500]))
                    sum_similar_loss = 0
                    num_similar_loss = 0
                    sum_dissimilar_loss = 0
                    num_dissimilar_loss = 0
                    ratios = np.zeros_like(ratios)
                    if additional_code_to_run is not None: additional_code_to_run(historic_data)
                if epochs_not_improved >= patience:
                    print("Not improved for %s epochs. Stopping..." % patience)
                    break
            print("[%s] Training Finished..." % time.asctime())
        except (InterruptedError, KeyboardInterrupt):
            print("Interrupted. Exiting training gracefully...")

        return best_score, historic_data

import numpy as np

from data.dataimport import import_data
from data.featuredict import FeatureDictionary


class TokenAutoencoderDatasetExtractor:
    SEQUENCE_START = "<s>"
    SEQUENCE_END = "</s>"

    def __init__(self, filename):
        training_data = import_data(filename)
        self.num_equivalence_classes = len(training_data)

        def vocabulary():
            for data in training_data.values():
                for token in self.__add_start_end_symbols(data["original"][0]):
                    yield token
                for noisy_sample in data["noise"]:
                    for token in self.__add_start_end_symbols(noisy_sample[0]):
                        yield token

        self.__feature_map = FeatureDictionary.get_feature_dictionary_for(vocabulary())

        dataset = self.build_dataset(training_data)
        self.__dataset = dataset

    @staticmethod
    def __add_start_end_symbols(sequence):
        return [TokenAutoencoderDatasetExtractor.SEQUENCE_START] + sequence + [
            TokenAutoencoderDatasetExtractor.SEQUENCE_END]

    def build_dataset(self, data):
        dataset = []
        for data in data.values():
            original = self.tokens_to_array(data["original"][0])

            noisy_samples = []
            for noisy_sample in data["noise"]:
                noisy_samples.append(self.tokens_to_array(noisy_sample[0]))
            dataset.append((original, noisy_samples))
        return dataset

    def tokens_to_array(self, tokens):
        return np.array([self.__feature_map.get_id_or_unk(t)
                         for t in self.__add_start_end_symbols(tokens)], dtype=np.int32)

    @staticmethod
    def get_pairs(dataset, use_noise=True, semantically_equivalent_noise=False):
        """
        Yield all pairs (original, noisy)
        """
        for code_snippet in dataset:
            original = code_snippet[0]
            yield original, original
            for noise in code_snippet[1]:
                yield original, noise
                if semantically_equivalent_noise:
                    yield noise, original

    def get_dataset_for_encoder(self, training_data, return_num_tokens=False):
        for idx, data in enumerate(training_data.values()):
            original_tokens = data["original"][0]
            original_num_toks = len(data["original"][0])
            noise_tokens = data["noise"]

            if return_num_tokens:
                yield self.tokens_to_array(original_tokens), original_num_toks, idx
            else:
                yield self.tokens_to_array(original_tokens), idx

            for noise in noise_tokens:
                noise_num_toks = len(noise[0])
                if return_num_tokens:
                    yield self.tokens_to_array(noise[0]), noise_num_toks, idx
                else:
                    yield self.tokens_to_array(noise[0]), idx

    @staticmethod
    def get_nonnoisy_samples(dataset):
        for code_snippet in dataset:
            yield code_snippet[0]

    @property
    def feature_map(self):
        return self.__feature_map

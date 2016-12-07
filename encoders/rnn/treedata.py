from collections import namedtuple

import numpy as np

from data.dataimport import import_data
from data.featuredict import FeatureDictionary, get_empirical_distribution
from data.tree import Node


class TreeDatasetExtractor:
    def __init__(self, filename: str, training_data: dict = None):
        """
        :param filename: the filename of the training data, ignored if training_data is *not* None
        :param training_data: use this training data instead of loading the filename, defaults to None (and thus
        data is loaded from the filename)
        """
        if training_data is None:
            training_data = import_data(filename)

        self.num_equivalent_classes = len(training_data)

        def get_vocabulary():
            for data in training_data.values():
                original_tree = data["original"][1]
                for node in original_tree:
                    yield node.name

        self.__node_type_dict = FeatureDictionary.get_feature_dictionary_for(get_vocabulary())

        def get_top_level_symbols():
            for data in training_data.values():
                original_tree = data["original"][1]
                yield original_tree.symbol
                for noise_expr in data["noise"]:
                    yield noise_expr[1].symbol

        self.__symbol_dict = FeatureDictionary.get_feature_dictionary_for(get_top_level_symbols(), 0)
        self.__empirical_symbol_dist = get_empirical_distribution(self.__symbol_dict, get_top_level_symbols())

        def get_num_properties():
            for data in training_data.values():
                original_tree = data["original"][1]
                for node in original_tree:
                    yield len(node.properties)

        self.__max_num_properties_per_node = max(get_num_properties())

        def get_start_node():
            for data in training_data.values():
                original_tree = data["original"][1]
                yield original_tree.name

        tree_roots = set(get_start_node())
        assert len(tree_roots) == 1  # Everything should be a block!
        self.__root_type = tree_roots.pop()

        self.__node_to_properties = {}
        for data in training_data.values():
            original_tree = data["original"][1]
            for node in original_tree:
                self.__node_to_properties[node.name] = node.properties

    def get_originals_from_dataset(self, data, return_num_tokens=False):
        for data in data.values():
            original_tree = data["original"][1]
            if return_num_tokens:
                yield self.convert_tree_to_array(original_tree), len(data["original"][0])
            else:
                yield self.convert_tree_to_array(original_tree)

    def get_node_properties(self, node_name):
        return self.__node_to_properties[node_name]

    @property
    def root_node_type(self):
        return self.__root_type

    @property
    def training_empirical_distribution(self):
        return self.__empirical_symbol_dist

    TreeArrayRepresentation = namedtuple("TreeArrayRepresentation", ['terminal_idx', 'terminal_types', 'current_idx',
                                                                     'children_idxs', 'node_types', 'num_nodes',
                                                                     'eq_symbol'])

    def convert_tree_to_array(self, tree: Node, return_node_to_id: bool = False, allow_any_root: bool = False,
                              ignore_eq_symbols=False) -> tuple:
        """
        Convert a tree to its array representation.
        """
        if not allow_any_root:
            assert tree.name == self.root_node_type, \
                "Tree does not have as root the start symbol (%s) but has: %s" % (self.root_node_type, tree.name)
        node_to_id_map = {}
        terminal_idxs = []
        terminal_types = []
        current_idx = []
        all_children_idxs = []
        node_types = []

        all_nodes_in_order = [n for n in tree][::-1]

        for i, node in enumerate(all_nodes_in_order):
            node_to_id_map[node] = i

            if len(node.properties) == 0:  # if is terminal
                terminal_idxs.append(i)
                terminal_types.append(self.__node_type_dict.get_id_or_unk(node.name))
            else:
                current_idx.append(i)
                node_types.append(self.__node_type_dict.get_id_or_unk(node.name))
                children_idxs = [-1] * self.max_num_properties_per_node
                for j, prp in enumerate(node.properties):
                    assert len(node[prp]) == 1, len(node[prp])
                    children_idxs[j] = node_to_id_map[node[prp][0]]
                all_children_idxs.append(children_idxs)

        if ignore_eq_symbols:
            tree_symbol = -1
        else:
            tree_symbol = self.__symbol_dict.get_id_or_none(tree.symbol)
            assert tree_symbol is not None

        converted_data = self.TreeArrayRepresentation(np.array(terminal_idxs, dtype=np.int32),
                                                      np.array(terminal_types, dtype=np.int32),
                                                      np.array(current_idx, dtype=np.int32),
                                                      np.array(all_children_idxs, dtype=np.int32),
                                                      np.array(node_types, dtype=np.int32), len(node_to_id_map),
                                                      np.array(tree_symbol, dtype=np.int32))

        if return_node_to_id:
            return converted_data, node_to_id_map
        else:
            return converted_data

    def get_dataset_for_semantic_similarity_encoder(self, training_data, return_num_tokens=False):
        for idx, data in enumerate(training_data.values()):
            original_tree = data["original"][1]
            original_converted = self.convert_tree_to_array(original_tree)
            original_tree_size = len(original_tree)
            noise_trees = data["noise"]

            if return_num_tokens:
                yield original_converted, original_tree_size, idx
            else:
                yield original_converted, idx

            for noise in noise_trees:
                assert original_tree.name == noise[1].name, (original_tree.name, noise[1].name)
                noise_converted_tree = self.convert_tree_to_array(noise[1])
                noise_tree_size = len(noise[1])
                if return_num_tokens:
                    yield noise_converted_tree, noise_tree_size, idx
                else:
                    yield noise_converted_tree, idx

    @property
    def max_num_properties_per_node(self):
        return self.__max_num_properties_per_node

    @property
    def node_type_dictionary(self):
        return self.__node_type_dict

    @property
    def symbol_dict(self):
        return self.__symbol_dict

    def get_dataset_for_encoder(self, training_data, return_num_tokens=False):
        for idx, data in enumerate(training_data.values()):
            original_tree = data["original"][1]
            original_converted = self.convert_tree_to_array(original_tree)
            original_num_toks = len(data["original"][0])
            noise_trees = data["noise"]

            if return_num_tokens:
                yield original_converted, original_num_toks, idx
            else:
                yield original_converted, idx

            for noise in noise_trees:
                assert original_tree.name == noise[1].name, (original_tree.name, noise[1].name)
                noise_converted_tree = self.convert_tree_to_array(noise[1])
                noise_num_toks = len(noise[0])
                if return_num_tokens:
                    yield noise_converted_tree, noise_num_toks, idx
                else:
                    yield noise_converted_tree, idx

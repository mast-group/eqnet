import itertools
import os
import pickle


class Node:
    """
    A node in an AST. Each node has a name and a fixed set of
    properties. For each property, the node may have zero or more
    children.
    """

    def __init__(self, name: str, properties: tuple, symbol: str = None, parent=None):
        """
        Initialize a node
        :param name:
        :param parent:
        :rtype parent: Node
        """
        self.__node_name = name
        self.__symbol = symbol
        self.__parent = parent
        self.__properties = tuple(properties)
        self.__children = {k: tuple() for k in properties}

    @property
    def name(self) -> str:
        return self.__node_name

    @property
    def symbol(self):
        return self.__symbol

    @property
    def parent(self):
        return self.__parent

    def __len__(self):
        return len([n for n in self])

    @property
    def depth(self):
        flat_children = list(itertools.chain(*self.__children.values()))
        if len(flat_children) == 0:
            return 0
        else:
            return max(c.depth for c in flat_children) + 1

    def parent_and_pos(self):
        """
        :return: return a tuple containing the parent node and the position (property and index) that this
         node appears in.
        """
        for property_name in self.__parent.properties:
            for i, child in enumerate(self.__parent[property_name]):
                if child == self:
                    return self.__parent, property_name, i
        assert False, "This must be impossible unless the tree was not constructed properly."

    @property
    def properties(self) -> tuple:
        """
        The node children properties
        """
        return self.__properties

    def set_children_for_property(self, property_name: str, children: iter):
        if property_name not in self.__properties:
            raise Exception(property_name + " not a property of this Node")
        self.__children[property_name] = tuple(children)

    def serialize(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(filename)

    def __getitem__(self, key: str):
        return self.__children[key]

    def __iter__(self):
        """
        Return a preorder traversal of the nodes in the subtree rooted at self.
        """
        to_visit = [self]

        while len(to_visit) > 0:
            current_node = to_visit.pop()
            yield current_node

            for property_name in current_node.properties[::-1]:
                to_visit.extend(current_node[property_name][::-1])

    def __str__(self):
        """
         Useful only for debugging purposes (should be slow)
        """
        return ''.join(self.__pretty_print())

    def __pretty_print(self, out_so_far=None, current_prefix="", is_last=False, annotator=None,
                       print_symbols: bool = True):
        if not out_so_far:
            out_so_far = []
        out_so_far.append(current_prefix)
        out_so_far.append("\-" if is_last else "-")
        out_so_far.append(str(self.name))
        if self.symbol is not None and print_symbols:
            out_so_far.append(" (" + str(self.symbol) + ")")
        if annotator is not None:
            out_so_far.append(" ")
            out_so_far.append(annotator(self))
        out_so_far.append(os.linesep)
        for i, property_name in enumerate(self.__properties):
            property_children = self.__children[property_name]
            out_so_far.append(current_prefix)
            is_last_property = i + 1 == len(self.__properties)
            out_so_far.append(("  " if is_last else "| ") + ("\+" if is_last_property else "|+"))
            out_so_far.append(str(property_name))
            out_so_far.append(os.linesep)
            prefix = current_prefix + ("  " if is_last else "| ") + (" " if is_last_property else "|")
            for j, child in enumerate(property_children):
                is_last_child = j + 1 == len(property_children)
                if is_last_child:  # is the last child
                    next_prefix = prefix + " "
                else:
                    next_prefix = prefix + " |"
                child.__pretty_print(out_so_far, next_prefix, is_last_child, annotator=annotator)
        return out_so_far

    def to_annotated_tree(self, annotator, print_symbols: bool = False):
        """
        :param annotator: a lambda from a Node to a string annotation
        """
        return ''.join(self.__pretty_print(annotator=annotator, print_symbols=print_symbols))

    @staticmethod
    def __maximal_common_subtree(tree1, tree2):
        assert tree1.name == tree2.name and tree1.properties == tree2.properties
        common_tree = Node(tree1.name, tree1.properties)
        to_check = [(tree1, tree2, common_tree)]
        while len(to_check) > 0:
            node1, node2, common_node = to_check.pop()
            for prop in node1.properties:
                ch1 = node1[prop]
                ch2 = node2[prop]
                if len(ch1) != len(ch2):
                    continue
                if all(c1.name == c2.name and c1.properties == c2.properties for c1, c2 in zip(ch1, ch2)):
                    new_children = [Node(c1.name, c1.properties, parent=common_node) for c1 in ch1]
                    common_node.set_children_for_property(prop, new_children)
                    to_check.extend(zip(ch1, ch2, new_children))
        return common_tree

    @staticmethod
    def _maximal_common_subtree(tree1, tree2):
        """Return the maximal common subtree rooted at these nodes"""
        current_maximal_tree = None
        current_maximal_tree_size = 0
        for node1 in tree1:
            for node2 in tree2:
                if node1.name == node2.name:
                    maximal_subtree_rooted_here = Node.__maximal_common_subtree(node1, node2)
                    size_of_tree = len(maximal_subtree_rooted_here)
                    if size_of_tree > current_maximal_tree_size:
                        current_maximal_tree_size = size_of_tree
                        current_maximal_tree = maximal_subtree_rooted_here
        return current_maximal_tree

    def maximal_common_subtree(self, other):
        return self._maximal_common_subtree(self, other)

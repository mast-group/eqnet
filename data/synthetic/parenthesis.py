import random
import sys
from collections import OrderedDict

from data.datasetgenerator import save_result_as_gzipped_json
from data.tree import Node


def synthesize_random_trees(stop_probability: float, max_nodes=500):
    assert 0 < stop_probability < 1
    available_nodes = [('()', ('child',)), ('[]', ('child',)), ('{}', ('child',)), ('<>', ('child',)), ]
    selected_node_type = available_nodes[random.randint(0, len(available_nodes) - 1)]
    seq_node_type = ('SEQ', ('child', 'next'))
    root = Node('Start', ('child',))

    num_generated = 1
    to_generate = [(root, 'child')]
    while len(to_generate) > 0:
        next_node, property_to_generate = to_generate.pop()
        p_stop = random.random()
        seq_selection = random.random()
        if (
                    p_stop < stop_probability and not next_node.name == 'SEQ' and not next_node.name == 'Start') or num_generated > max_nodes:
            next_node.set_children_for_property(property_to_generate, (Node('Empty', (), parent=next_node)))
            continue
        num_generated += 1
        current_node_type = selected_node_type
        if seq_selection < 0.25:
            current_node_type = seq_node_type
        child = Node(current_node_type[0], current_node_type[1], parent=next_node)
        next_node.set_children_for_property(property_to_generate, (child,))
        for child_property in current_node_type[1]:
            to_generate.append((child, child_property))
    return root


def to_token_sequence(tree: Node, current_tokens: list) -> list:
    if tree.name == 'Start':
        to_token_sequence(tree['child'][0], current_tokens)
    elif tree.name == 'Empty':
        return
    elif tree.name == 'SEQ':
        to_token_sequence(tree['child'][0], current_tokens)
        to_token_sequence(tree['next'][0], current_tokens)
    else:
        name = tree.name
        current_tokens.append(name[0])
        to_token_sequence(tree['child'][0], current_tokens)
        current_tokens.append(name[1])
    return current_tokens


def convert_to_dict(node: Node) -> dict:
    children = OrderedDict()
    for node_property in node.properties:
        children[node_property] = convert_to_dict(node[node_property][0])
    if len(children) > 0:
        return dict(Name=node.name, Children=children)
    else:
        return dict(Name=node.name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage <outputFilename>")
        sys.exit(-1)
    samples = {}
    for i in range(100000):
        tree = synthesize_random_trees(.75)
        toks = to_token_sequence(tree, [])
        if len(
            toks) > 500: continue  # Too large sequences probably cause problems to Theano (or require too much memory?)
        asstring = ''.join(toks)
        samples[asstring] = (toks, tree)
    print("Generated %s samples" % len(samples))

    converted_to_standard_format = {}
    for n, t in samples.items():
        converted_to_standard_format[n] = dict(Original=dict(Tokens=t[0], Tree=convert_to_dict(t[1])), Noise=[])

    save_result_as_gzipped_json(sys.argv[1], converted_to_standard_format)

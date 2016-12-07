import gzip
import json
from collections import OrderedDict

from data.tree import Node


def parse_ast(ast_data):
    """
    Given an AST convert it to Node representation.
    :param ast_data:
    :return:
    """
    root_node = Node(ast_data["Name"],
                     properties=tuple(p for p in ast_data["Children"]) if "Children" in ast_data else (),
                     symbol=None if "Symbol" not in ast_data else ast_data["Symbol"])
    to_visit = [(ast_data, root_node)]

    while len(to_visit) > 0:
        node_data, current_node = to_visit.pop()
        if "Children" not in node_data: continue
        for property_name, child_node in node_data["Children"].items():
            node = Node(child_node["Name"],
                        properties=tuple(p for p in child_node["Children"]) if "Children" in child_node else (),
                        symbol=None if "Symbol" not in child_node else child_node["Symbol"],
                        parent=current_node)
            current_node.set_children_for_property(property_name, node)
            to_visit.append((child_node, node))

    return root_node


def import_data(filename: str) -> dict:
    """
    Import data from C#
    :param filename:
    :return: a dictionary (fully_qualified_method_name-> (token_stream, ast))
    """
    with gzip.open(filename, 'rb') as f:
        data = json.loads(f.read().decode('utf-8'), object_pairs_hook=OrderedDict)

    imported_data = {}
    for method_name, code_data in data.items():
        noisy_samples = []
        for sample in code_data["Noise"]:
            tokens = sample["Tokens"]
            ast_data = parse_ast(sample["Tree"])
            noisy_samples.append((tokens, ast_data))

        original_tokens = code_data["Original"]["Tokens"]
        original_ast_data = code_data["Original"]["Tree"]
        imported_data[method_name] = {"original": (original_tokens, parse_ast(original_ast_data)),
                                      "noise": noisy_samples}
    return imported_data


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage <inputFile>")
        sys.exit(-1)
    import_data(sys.argv[1])

import sys
from collections import defaultdict
from math import log2

from data.datasetgenerator import open_gzipped_json_data

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage <dataset>")
        sys.exit(-1)

    # Load JSON
    input_filename = sys.argv[1]
    assert input_filename.endswith('.json.gz')
    all_data = open_gzipped_json_data(input_filename)

    # Put equivalence classes into buckets
    equivalence_classes = defaultdict(list)

    for eq_class, elements in all_data.items():
        equivalence_classes[eq_class] = [elements['Original']] + elements['Noise']

    total_num_expressions = sum(len(e) for e in equivalence_classes.values())
    print([len(e) for e in equivalence_classes.values()])
    print("Num Equivalence Classes: %s" % len(equivalence_classes))
    print("Total num expressions: %s" % total_num_expressions)
    print("Avg Num of Expressions per Eq Class: %s" % (float(total_num_expressions) / len(equivalence_classes)))

    entropy_elements = ((float(len(e)) / total_num_expressions) for e in equivalence_classes.values())
    entropy = sum(e * log2(e) for e in entropy_elements)
    print("Entropy (bits): %s" % entropy)

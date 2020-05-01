import random
import string

import numpy as np
import onnx
from onnx import numpy_helper

import argparse


def main(input_model_name: str):
    model = onnx.load_model(input_model_name)

    # Ensure initial GraphProto is valid
    onnx.checker.check_model(model)
    print("Input model passed ONNX validation")

    rename_mapping = dict()
    new_data = []

    # Create a whole new set of TensorProtos with random names
    for tensor_proto in model.graph.initializer:
        numpy_array = numpy_helper.to_array(tensor_proto)
        rand_name = ''.join(
            random.choices(string.ascii_uppercase + string.digits, k=10))
        new_data.append(
            numpy_helper.from_array(np.zeros_like(numpy_array),
                                    name=rand_name))
        rename_mapping[tensor_proto.name] = rand_name

    # Replace old data with new data
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_data)

    # Run over graph pointing to the new zero'd out data
    def rewrite_names(repeated_field):
        for proto in repeated_field:
            if proto.name in rename_mapping:
                proto.name = rename_mapping[proto.name]

    rewrite_names(model.graph.input)
    rewrite_names(model.graph.output)
    rewrite_names(model.graph.value_info)
    rewrite_names(model.graph.node)

    for node in model.graph.node:
        for j, name in enumerate(node.input):
            if name in rename_mapping:
                node.input[j] = rename_mapping[name]
        for j, name in enumerate(node.output):
            if name in rename_mapping:
                node.input[j] = rename_mapping[name]

    # Ensure final GraphProto is valid
    print("Making sure output ONNX model is valid")
    onnx.checker.check_model(model)
    print("Output model passed ONNX validation, writing to disk")
    onnx.save(model, input_model_name + '.zero')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_model', help='input onnx model')

    args = parser.parse_args()
    main(args.input_model)
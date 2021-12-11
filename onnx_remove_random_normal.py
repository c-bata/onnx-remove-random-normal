import argparse
import copy
import struct
from typing import List

import numpy as np
import onnx
from onnx.onnx_ml_pb2 import ModelProto, NodeProto, AttributeProto, TensorProto

parser = argparse.ArgumentParser(description='Replace ONNX RandomNormal node with Constant Node')
parser.add_argument('source', help='source onnx file')
parser.add_argument('output', help='output onnx file')
parser.add_argument('--seed', type=int, default=None, help='random seed')
args = parser.parse_args()

rnd = np.random.RandomState(args.seed)


def get_shape(node: NodeProto) -> List[int]:
    for attr in node.attribute:
        if attr.name == "shape" and \
                attr.type == AttributeProto.AttributeType.INTS:
            return list(attr.ints)
    raise RuntimeError("shape is not found")


def get_float_values_from_bytes(x: bytes) -> List[float]:
    floats: List[float] = []
    for i in range(len(x) // 4):
        # little-endian float32 (4 bytes)
        floats.append(struct.unpack("<f", x[i*4:(i+1)*4])[0])
    return floats


def replace_random_normal_with_constant(model: ModelProto) -> ModelProto:
    onnx.checker.check_model(model)

    copied = copy.deepcopy(model)
    for i, node in enumerate(model.graph.node):
        if node.op_type == "RandomNormal":
            new_node = NodeProto(
                name=f"Constant_From_{node.name}",  # set unique name
                op_type="Constant",
                output=node.output,
                doc_string="This node is replaced from RandomNormal.\n\n" + node.doc_string
            )

            shape = get_shape(node)
            # When this raw_data field is used to store tensor value, elements MUST
            # be stored in as fixed-width, little-endian order.
            # https://github.com/onnx/onnx/blob/b9972345/onnx/onnx-ml.proto#L576-L577
            #
            # >>> x = np.random.randn(2).astype('float32')
            # >>> x
            # array([ 1.24379277, -1.09057736], dtype=float32)
            # >>> struct.unpack("<f", bytearray(x)[:4])
            # (1.2437927722930908,)
            constant_values = rnd.randn(*shape).astype('float32')
            new_node.attribute.append(
                AttributeProto(
                    name="value",
                    type=AttributeProto.AttributeType.TENSOR,
                    t=TensorProto(
                        dims=shape,
                        data_type=TensorProto.DataType.FLOAT,
                        raw_data=bytes(bytearray(constant_values))
                    )
                )
            )
            copied.graph.node.pop(i)
            copied.graph.node.insert(i, new_node)
    onnx.checker.check_model(copied)
    return copied


def main():
    model = onnx.load(args.source)
    output = replace_random_normal_with_constant(model)
    with open(args.output, "wb") as f:
        f.write(output.SerializeToString())


if __name__ == '__main__':
    main()

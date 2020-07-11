import numpy as np

from mo.graph.graph import Graph, Node
from mo.ops.op import Op
import logging as log
from mo.front.common.partial_infer.elemental import copy_shape_infer

class Placeholder(Op):

    op = 'Placeholder'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': 'Placeholder',

            'infer': self.infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        copy_shape_infer(node, lambda n: n.in_node().value.astype(n.dst_type))
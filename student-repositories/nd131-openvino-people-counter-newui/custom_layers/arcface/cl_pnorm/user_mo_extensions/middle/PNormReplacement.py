"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import networkx as nx
import numpy as np

from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from extensions.ops.ReduceOps import ReduceMean
from extensions.ops.merge import Merge
from extensions.ops.Cast import Cast
from extensions.ops.activation_ops import Floor, Sigmoid
from extensions.ops.exp import ExpOp
from extensions.ops.elementwise import Mul, Div, Pow, Elementwise
from mo.ops.reshape import Reshape
from mo.ops.concat import Concat
from mo.front.common.partial_infer.utils import int64_array, float_array
from ..utils import reduce_infer
import math
from mo.ops.power import Power
import logging as log
from ..front.map import *

class PNormMiddleReplacement(MiddleReplacementPattern):
    op = 'Merge'
    enabled = True
    replacement_id = "ObjectDetectionAPIPreprocessorReplacement"
    force_clean_up = False

    # run passes after
    def run_after(self):
        return []

    # run passes before
    def run_before(self):
        from .L2NormToNorm import L2NormToNormPattern
        from extensions.middle.pass_separator import MiddleStart, MiddleFinish
        # spatial axis = 0, shape is defined to be B 1 1 5
        return [L2NormToNormPattern]

    @staticmethod
    def pattern(**kwargs):
        # l2_normalize_data, l2_normalize
        return dict(
            nodes=[
                ('merge', dict(op='Merge', name='batch_normalization_11/cond/Merge'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):

        const = 0.99
        merge = match['merge']
        digits = significant_digits()
        pnorm = Power(graph, {
            'name': merge.name + '/reciprocal_', 'type': 'PNORM',
            'significant': digits[0], 'to_significant': digits[1],
            'scale': 1, 'shift': 0, 'power': get_power_attr()
        }).create_node()
        merge.in_port(0).get_connection().set_destination(pnorm.in_port(0))

        in_shape = pnorm.in_port(0).data.get_shape()
        in_shape = list(in_shape)
        in_shape.insert(0, 1)

        reshape1 = Reshape(graph, {'name': merge.name + '/Reshape_Node1'}).create_node()
        reshape_dim1 = Const(graph, {'value': np.array(in_shape), 'name': merge.id + '/Reshape_Dim1'}).create_node()
        pnorm.out_port(0).connect(reshape1.in_port(0))
        reshape1.in_port(1).connect(reshape_dim1.out_port(0))

        concat_node = Concat(graph, {'axis': 0, 'name': merge.name + '/Concat_', 'override_output_shape': True}).create_node()
        const3 = Const(graph, {'name': merge.name + '/const_reduce', 'value': 0}).create_node()

        for ii, idx in enumerate(range(pnorm.significant, pnorm.to_significant+1, 1)):
            const_node = Const(graph, 
            {'value': float_array(math.pow(const,idx)),
            'name': merge.name + '/Const_'+ii.__str__()}).create_node()

            mul_node = Mul(graph, {'name': merge.name + '/Mul_'+ii.__str__()}).create_node()
            const_node.out_port(0).connect(mul_node.in_port(0))
            
            reshape1.out_port(0).connect(mul_node.in_port(1)) # connect to the graph node
            mul_node2 = Mul(graph, {'name': merge.name + '/Mul_Div_'+ii.__str__()}).create_node()
            
            const_node2 = Const(graph, 
            {'value': float_array(math.pow(const,-1*idx)),
            'name': merge.name + '/Const_Pow_'+ii.__str__()}).create_node()
            cast_node = ExpOp(graph, 
            {'name': merge.name + '/Exp_'+idx.__str__()}).create_node()
            
            mul_node.out_port(0).connect(cast_node.in_port(0))
            const_node2.out_port(0).connect(mul_node2.in_port(1))
            cast_node.out_port(0).connect(mul_node2.in_port(0))
            concat_node.add_input_port(ii, skip_if_exist=True)
            concat_node.in_port(ii).get_connection().set_source(mul_node2.out_port(0))

        in_shape = pnorm.in_port(0).data.get_shape()
        in_shape = list(in_shape)
        
        reducesum_node = ReduceMean(graph, 
        {'name': merge.id + '/_pnorm_reduced_sum', 
        'keep_dims': True, 'in_ports_count': 2, 
        'shape': in_shape, 'axis': 0,
        'need_shape_inference': None, 'infer': reduce_infer}).create_node()

        const3.out_port(0).connect(reducesum_node.in_port(1))
        reducesum_node.in_port(0).get_connection().set_source(concat_node.out_port(0))

        reshape = Reshape(graph, {'name': merge.name + '/Reshape_Node'}).create_node()
        reshape_dim = Const(graph, {'value': np.array(in_shape), 'name': merge.id + '/Reshape_Dim'}).create_node()
        reducesum_node.out_port(0).connect(reshape.in_port(0))
        reshape.in_port(1).connect(reshape_dim.out_port(0))
        merge.out_port(0).get_connection().set_source(reshape.out_port(0))

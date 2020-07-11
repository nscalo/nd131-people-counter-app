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
from mo.front.common.partial_infer.utils import int64_array, float_array
import math
from mo.ops.concat import Concat
from extensions.ops.activation_ops import Activation
import logging as log

class LOGNormMiddleReplacement(MiddleReplacementPattern):
    op = 'FullyConnected'
    enabled = True
    replacement_id = "ObjectDetectionAPIPreprocessorReplacement"
    force_clean_up = False

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    @staticmethod
    def pattern(**kwargs):
        # l2_normalize_data, l2_normalize
        return dict(
            nodes=[
                ('fc', dict(name='dense_1/MatMul'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):

        merge = match['fc']
        Activation.operation = staticmethod(lambda x: x)
        
        def supported_attrs(self):
            return ['scale']
        
        Activation.supported_attrs = supported_attrs
        activation = Activation(graph, 
        dict(name=merge.id + '/lognorm_', 
        scale=1.0, type='LOGNORM')).create_node()

        merge.in_port(0).get_connection().set_destination(activation.in_port(0))
        activation.out_port(0).connect(merge.in_port(0))
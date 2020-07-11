#!/opt/intel/openvino_2019.3.376/deployment_tools/model_optimizer/venv/bin/python3
""" Cog content generation tool.
    http://nedbatchelder.com/code/cog

    Copyright 2004-2019, Ned Batchelder.
"""

import sys
from cogapp import Cog

sys.exit(Cog().main(sys.argv))

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging
import sklearn
import data

# Set PATHs
# path to senteval
PATH_TO_SENTEVAL = '../'
# path to the NLP datasets 
PATH_TO_DATA = '../data/downstream'
# path to glove embeddings
PATH_TO_VEC = '../pretrained/glove.840B.300d.txt'


# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


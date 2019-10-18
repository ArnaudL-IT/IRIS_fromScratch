"""
Multiclass classifier for the IRIS dataset
implemented from scratch (i.e. without libraries such as scikit-learn)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

with open('./data/iris.dat') as file:
    df = pd.read_csv(file, header = None)

"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import os
import torch.optim as optim
from dataloader import load_dataset, decode_one_seq
import glob
import numpy as np
from sklearn.preprocessing import OneHotEncoder

    # Function definitions

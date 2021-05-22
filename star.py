
from itertools import dropwhile
import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataframe = pd.read_csv('star.csv')

dataframe.head()
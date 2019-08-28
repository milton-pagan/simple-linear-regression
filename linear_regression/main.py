import pandas as pd
import matplotlib.pyplot as plt
from grad_descent import *

# Import datasets

dta_set1 = pd.read_csv("/home/milton/Documents/machine-learning-ex1/ex1/ex1data1.txt")
dta_set2 = pd.read_csv("/home/milton/Documents/machine-learning-ex1/ex1/ex1data2.txt")

dta_set1.insert(0, 'X_0', np.ones(len(dta_set1.index)))
dta_set1.columns = ['X_0','X_1','Y']

dta_set2.insert(0, 'X_0', np.ones(len(dta_set2.index)))
dta_set2.columns = ['X_0','X_1', 'X_2','Y']

grad_descent = GradDescent(0.02)

grad_descent.run(dta_set1)

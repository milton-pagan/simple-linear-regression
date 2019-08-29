import pandas as pd
import numpy as np
from linear_regression.grad_descent import *
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

# Import datasets

dta_set1 = pd.read_csv("/home/milton/Documents/machine-learning-ex1/ex1/ex1data1.txt")
dta_set2 = pd.read_csv("/home/milton/Documents/machine-learning-ex1/ex1/ex1data2.txt")

# Add X_0 columns
dta_set1.insert(0, 'X_0', np.ones(len(dta_set1.index)))
dta_set1.columns = ['X_0', 'X_1', 'Y']

dta_set2.insert(0, 'X_0', np.ones(len(dta_set2.index)))
dta_set2.columns = ['X_0', 'X_1', 'X_2', 'Y']

grad_descent = GradDescent(dta_set1, 0.02)

grad_descent.run()

plt.figure(1)

plt.title("One dimensional linear regression")

plt.scatter(dta_set1['X_1'].to_numpy(), dta_set1['Y'].to_numpy())

x = np.linspace(dta_set1['X_1'].min(), dta_set1['X_1'].max(), len(dta_set1.index))

plt.plot(x, [(grad_descent.parameters[1] * i + grad_descent.parameters[0]) for i in x], color='brown')

plt.show()

grad_descent.learning_rate = 0.9
grad_descent.change_dataset(dta_set2)
grad_descent.mean_normalize()
grad_descent.run()

fig = plt.figure(2)
ax = plt.axes(projection='3d')

plt.title("Two dimensional linear regression")

ax.scatter3D(dta_set2['X_1'].to_numpy(), dta_set2['X_2'].to_numpy(), dta_set2['Y'].to_numpy(), c=dta_set2['Y'].to_numpy())

x = np.linspace(dta_set2['X_2'].min(), dta_set2['X_2'].max(), len(dta_set2.index))
y = np.linspace(dta_set2['X_1'].min(), dta_set2['X_1'].max(), len(dta_set2.index))

X, Y = np.meshgrid(x, y)

Z = np.array([(grad_descent.parameters[2] * x + grad_descent.parameters[1] * y + grad_descent.parameters[0]) for x, y in zip(X, Y)])

ax.plot_wireframe(X, Y, Z, color='green')

plt.show()

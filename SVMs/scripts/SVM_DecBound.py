import sys
if sys.version_info[0] < 3:
	raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy import io

original = np.load('./data/toy-data.npz')
data = original['training_data']

w = np.array([-0.4528, -0.5190])
b = 0.1471
labels = original['training_labels']

plt.scatter(data[:, 0], data[:, 1], c=labels)

#Plotthedecisionboundary 
x = np.linspace(-5,5,100) 
y=-(w[0]*x+b)/w[1] 
plt.plot(x,y, 'k', label = "Decision Boundary") 

#Plotthemargins
upper = -(w[0]*x+b - 1)/w[1] 
plt.plot(x, upper, label = "Upper Margin")

lower = -(w[0]*x+b + 1)/w[1] 
plt.plot(x, lower, label = "Lower Margin")

plt.legend()
plt.show()
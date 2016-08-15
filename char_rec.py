import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import time
import warnings
import cv2

warnings.filterwarnings("ignore")

digits=datasets.load_digits()

clf=svm.SVC(gamma=0.02)

# get the inputs and outputs
input,output=digits.data[:-10] , digits.target[0:-10]

# now load them in and train the classifier machine
clf.fit(input,output)

print"Training done"



# once trained lets test it


# print clf.predict(digits.data[-6])
print("The predicted number is"),
print clf.predict(digits.data[-7])
# plt.imshow(digits.images[-6], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()
# time.sleep(1)
plt.imshow(digits.images[-7], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

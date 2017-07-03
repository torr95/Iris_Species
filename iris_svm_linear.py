#support vector machine with linear kernel used instead of deciosn tree or naive_bayes.

from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
import random
from sklearn.metrics import accuracy_score

iris = load_iris()

test_idx=[random.randrange(1,150,1,int) for i in range(30)]
print(test_idx)

train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data, test_idx, axis= 0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = SVC(kernel='linear')
clf.fit(train_data,train_target)

predict=clf.predict(test_data)

print("original_result",test_target)
print("predicted_result",predict)

print(accuracy_score(predict,test_target))


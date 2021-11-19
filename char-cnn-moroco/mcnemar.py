from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

labels_1 = np.loadtxt('pred-se-relu.txt')
labels_2 = np.loadtxt('pred-se-pynada.txt')
gt_labels = np.loadtxt('test_labels.txt')

binary_1 = (labels_1 == gt_labels) * 1
binary_2 = (labels_2 == gt_labels) * 1

conf_matrix = confusion_matrix(binary_1, binary_2)
print(conf_matrix)
print(mcnemar(conf_matrix, exact=False, correction=False))
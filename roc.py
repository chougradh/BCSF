import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle

from keras.utils import np_utils
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import dataset
import net

np.random.seed(1337)

n = 224
batch_size = 128
cvscores = []
data_directory, = sys.argv[1:]

X, y, tags = dataset.dataset(data_directory, n)
nb_classes = len(tags)

sample_count = len(y)

X_test  = X
y_test  = y
Y_test = np_utils.to_categorical(y_test, nb_classes)


model, tags_from_model = net.load("model")
assert tags == tags_from_model
net.compile(model)

y_score = model.predict(X_test, batch_size=batch_size)


colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2


# Compute ROC curve and ROC area 
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(nb_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:,i], y_score[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
plt.plot(fpr[1], tpr[1], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('roc')
plt.show()



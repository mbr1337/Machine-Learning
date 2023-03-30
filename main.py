import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score

data = pd.read_csv('diabetes.csv')
head = list(data.columns.values)
data.head()
data.info()

plt.hist(data['Outcome'])
plt.ylabel('Number of people')
plt.xlabel('Outcome')
plt.show()

# # zad5
# def scatter_plot(x, y):
#     red = data[data['Outcome'] == 1]
#     blue = data[data['Outcome'] == 0]
#     plt.scatter(red[x], red[y], color="red")
#     plt.scatter(blue[x], blue[y], color="blue")
#     plt.show()
#
#
# scatter_plot('BloodPressure', 'Insulin')

x1 = data.loc[data['Outcome'] == 0]
x2 = data.loc[data['Outcome'] == 1]

# for i in head:
#         for j in head:
#             if i!='Outcome' and j!='Outcome' and i!=j:
#                 data.plot.scatter(x=i,y=j,c='Outcome',colormap='viridis')
# plt.show()

x1['Insulin'].hist()
x1['BloodPressure'].hist()
plt.show()

x = data.loc[:, 'Pregnancies': 'Age']
y = data.loc[:, 'Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(x, y, shuffle=False)
mlp = MLPClassifier(hidden_layer_sizes=16, random_state=1, max_iter=900).fit(X_train, Y_train)

print(mlp.score(X_test, Y_test))





y_pred = mlp.predict(X_test)
print(y_pred)
# print(Y_test)

confusion_matrix = confusion_matrix(y_pred, Y_test)

print(confusion_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot()
plt.show()

# zad12
# In the multi-class and multi-label case, this is
# the average of the F1 score of each class with weighting
# depending on the average parameter.
F1 = f1_score(Y_test, y_pred)
print(F1)


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

animal_test = pd.read_csv("animals_test.csv")
animal_train = pd.read_csv("animals_train.csv")
animal_class = pd.read_csv("animal_classes.csv")


print("----Creating dictionary with Class id as Key and Class Name as value---------")
# Making a dictionary of animal_classes Key(num), Value(class_name)
# print(animal_class.head(3))
# dic={}
dic = pd.Series(animal_class.Class_Type.values,
                index=animal_class.Class_Number).to_dict()
# print(dic)

print('----Setting Output terminal Specs----')
pd.set_option("precision", 4)
pd.set_option('max_columns', 18)
pd.set_option('display.width', None)
#print(animal_train.head(3))

print('----Assigning Train and Test variables------')
train_data = animal_train[animal_train.columns.difference(['class_number'])]
train_label = animal_train[['class_number']]
test_data = animal_test[animal_test.columns.difference(['animal_name'])]

#train_data = animal_train.to_numpy()[:, :-1]
#train_label = animal_train.to_numpy()[:, -1]
#test_data = animal_test.to_numpy()[:, 1:]

print("----Running KNN Cluster Classifier----")
knn = KNeighborsClassifier()
knn.fit(X=train_data, y=train_label)
predicted = knn.predict(test_data)

print("---Round the Prediction---")
#print(predicted)
final_prediction = predicted.round(0).astype(int).flatten()

#print(final_prediction[:5])
#for i, r in animal_test[['animal_name']].iterrows():
 #   print(f'{r.animal_name}, {dic[final_prediction[i]]}')

print('---Create file with animal name and class----')
with open('classprediction.txt', 'w') as new_book:
    new_book.write("Animal Name,Animal Class\n")
    for i, r in animal_test[['animal_name']].iterrows():
        new_book.write(f'{r.animal_name}, {dic[final_prediction[i]]}\n')

print('done')

#Self Note Linear Regression
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#linear_regression = LinearRegression()
#linear_regression.fit(X=train_data, y=train_label)

# for i, name in enumerate(animal_train):
#     print(f"{name}:{linear_regression.coef_[i]}")
#predicted = linear_regression.predict(test_data)
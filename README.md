# Goal of the Project: Classification of News with Machine Learning Model: 

# In the era of internet and social media we have surrounded by n number of news. We are always searching for our favorite categories of News. Most of the News Websites, News editors are doing manual categorization of News into the different classes, Such as Sports, Entertainment, Business, Politics, Tech world..etc.. This takes long time and having threats of incorrect categorization. So, here is Machine Learning Algorithms which can help us to do this process automated. We will see how the Machine Learning would help us to News Categorization.

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier

df = pd.read_csv('NewsDataFile.csv')

df.head()

df.isnull().sum()

df.shape

df.columns

df.info()

cf  = df["category"].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,8))
plt.pie(cf)

plt.legend(["sport","business", "politics", "tech","entertainment"])
plt.show

cf

x = df["title"]
x

x.shape

y = df["category"]
y

y.shape

cv = CountVectorizer()

X_data = cv.fit_transform(x)

print(X_data)

x_arr = np.array(df["title"])
y_arr = np.array(df["category"])

x_arr

y_arr

X_arr_data = cv.fit_transform(x_arr)

print(X_arr_data)

BNB_arr = BernoulliNB()
BNB_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= BNB_arr.predict(x_arr_test)
BNB_arr.score(x_arr_test, y_arr_test)

RFC_arr = RandomForestClassifier()
RFC_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= RFC_arr.predict(x_arr_test)
RFC_arr.score(x_arr_test, y_arr_test)

RFC_arr = RandomForestClassifier(n_estimators=200, criterion='entropy')
RFC_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= RFC_arr.predict(x_arr_test)
RFC_arr.score(x_arr_test, y_arr_test)

SVC_arr = SVC()
SVC_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= SVC_arr.predict(x_arr_test)
SVC_arr.score(x_arr_test, y_arr_test)

SVC_arr = SVC(C=1.0,kernel = 'linear', gamma='auto')
SVC_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= SVC_arr.predict(x_arr_test)
SVC_arr.score(x_arr_test, y_arr_test)

DTC_arr = DecisionTreeClassifier()
DTC_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= DTC_arr.predict(x_arr_test)
DTC_arr.score(x_arr_test, y_arr_test)

DTC_arr = DecisionTreeClassifier(criterion = "entropy")
DTC_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= DTC_arr.predict(x_arr_test)
DTC_arr.score(x_arr_test, y_arr_test)

XGBC_arr = xgb.XGBClassifier()
XGBC_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= XGBC_arr.predict(x_arr_test)
XGBC_arr.score(x_arr_test, y_arr_test)

XGBC_arr = xgb.XGBClassifier(n_estimators = 100, gamma = 0.2)
XGBC_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= XGBC_arr.predict(x_arr_test)
XGBC_arr.score(x_arr_test, y_arr_test)


ABC_arr = AdaBoostClassifier()
ABC_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= ABC_arr.predict(x_arr_test)
ABC_arr.score(x_arr_test, y_arr_test)

ABC_arr = AdaBoostClassifier( n_estimators=150, learning_rate=1.0, algorithm='SAMME.R')
ABC_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= ABC_arr.predict(x_arr_test)
ABC_arr.score(x_arr_test, y_arr_test)

MNB.score(x_test, y_test)

x_arr_train, x_arr_test, y_arr_train, y_arr_test = train_test_split(X_arr_data, y_arr, test_size=0.2, random_state=51)

print("Shape of x_arr_train>> ",x_arr_train.shape)
print("Shape of y_arr_train>> ",y_arr_train.shape)
print("Shape of x_arr_test>> ",x_arr_test.shape)
print("Shape of y_arr_test>> ",y_arr_test.shape)

MNB_arr = MultinomialNB()

MNB_arr.fit(x_arr_train,y_arr_train)

y_arr_pred= MNB_arr.predict(x_arr_test)

cm_arr = confusion_matrix(y_arr_test, y_arr_pred, labels=MNB_arr.classes_)
disp_arr = ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=MNB_arr.classes_)

disp_arr.plot(xticks_rotation='vertical')
plt.figure(figsize=(10,20))
plt.show()

MNB_arr.score(x_arr_test, y_arr_test)

MNB_arr = MultinomialNB(alpha = 5.0)
MNB_arr.fit(x_arr_train,y_arr_train)
y_arr_pred= MNB_arr.predict(x_arr_test)
cm_arr = confusion_matrix(y_arr_test, y_arr_pred, labels=MNB_arr.classes_)
disp_arr = ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=MNB_arr.classes_)
disp_arr.plot(xticks_rotation='vertical')
plt.figure(figsize=(10,20))
plt.show()

MNB_arr.score(x_arr_test, y_arr_test)

x_train, x_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=51)

print(x_train)

y_train

print("Shape of x_train>> ",x_train.shape)
print("Shape of y_train>> ",y_train.shape)
print("Shape of x_test>> ",x_test.shape)
print("Shape of y_test>> ",y_test.shape)

MNB = MultinomialNB()

MNB.fit(x_train,y_train)

y_pred= MNB.predict(x_test)

y_pred

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred, labels=["sport","business", "politics", "tech","entertainment"]))

from sklearn.metrics import ConfusionMatrixDisplay


ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=MNB.classes_)


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=MNB.classes_)


disp.plot(xticks_rotation='vertical')
plt.figure(figsize=(10,20))
plt.show()

userdata = input("Enter a Text: ")
inputdata = cv.transform([userdata]).toarray()
output = MNB.predict(inputdata)
print(output)

# Conclusion: Project Summary:  


# Here we got the magic of automated categorization of News in multiple classes. We just need to select the correct categories to have a look on the favorite classes. Even though New Websites , News Editors or News Publishers can use this automatic machine learning technique to get categorization. Note: Please note, this Machine Learning algorithm is only for Learning Purpose.
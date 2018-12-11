# Authors: James Finn and Madina Martazanova
# This program is intended to apply a Naive Bayes Classifier  to a dataset containing information about phishing
#  websites.
# We will be using the Gaussian Naive Bayes classifier.

# Imports
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from utilities import visualize_classifier

if __name__ == '__main__':
    input_file = './phpV5QYya.csv'
	data = np.loadtxt(input_file, delimiter=',', skiprows=1)
	columns, classes = data[:, :-1], data[:, -1]

	columns_train, columns_test, classes_train, classes_test = model_selection.train_test_split(columns, classes, test_size=0.2)
#Create Naive Bayes classifier
	gnb = GaussianNB()
#Train the classifier
	gnb.fit(columns_train, classes_train)
#Predict the values for training data
	classes_pred = gnb.predict(columns_test)
#Compute accuracy of the classifier
	accuracy = 100.0 * (classes_test == classes_pred).sum() / columns_test.shape[0]
	print("Accuracy of innate classifier: " + str(round(accuracy, 2)) + "%")

	folds = 50
	cross_acc = model_selection.cross_val_score(gnb, columns, classes, scoring='accuracy', cv=folds)
	print("Cross Val accuracy: " + str(round(100*cross_acc.mean(), 2)) + "%")

	cross_pre = model_selection.cross_val_score(gnb, columns, classes, scoring='precision_weighted', cv=folds)
	print("Precision: " + str(round(100*cross_pre.mean(), 2)) + "%")

	f1_values = model_selection.cross_val_score(gnb, columns, classes, scoring='f1_weighted', cv=folds)
	print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")

	confusionmatrix = confusion_matrix(classes_test, classes_pred)
	print(confusionmatrix)

# Print classification report
	print("\n" + "#"*15 + " Bayes Results "+ "#"*15)
	print(classification_report(classes_test, classes_pred))
	print("\n" + "#"*40)

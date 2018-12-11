# Authors: James Finn and Madina Martazanova
# We will build a Support Vector Machine classifier to predict phishing attacks based on our dataset.
# We will be using '--classifier OneVsOneClassifier'

# Imports
import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import model_selection

if __name__ == '__main__':
# Load input file containing data
    input_file = '/home/jimmy/Documents/Brockport/CSC434 AI/FinalProject/phpV5QYya.csv'
    data = np.loadtxt(input_file, delimiter=',',skiprows=1)
    columns, classes = data[:, :-1], data[:, -1]

# Create SVM classifier with a linear kernel
    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
# Train the classifier
    classifier.fit(columns, classes)
# Perform model selection using model selection train
    columns_train, columns_test, classes_train, classes_test = \
        model_selection.train_test_split(columns, classes, test_size=0.20)
    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    classifier.fit(columns_train, classes_train)
# Predict the output using classifier.
    classes_test_pred = classifier.predict(columns_test)
# Print classification report
    print("\n" + "#"*15 + " SVM Results " + "#"*15)
    print(classification_report(classes_test, classes_test_pred))
    print("\n" + "#"*40)

# Authors: James Finn and Madina Martazanova
# This program is intended to apply a Random Forest Classifier to a dataset containing information about phishing
#   websites. The program expects a single command line argument, either '--classifier randomforest' or
#   '--classifier extremerandomforest', depending on which variety of Random Forest classifier is desired.

# Imports
import argparse  # used for parsing the command line arguments
import numpy as np  # used for ease in manipulating data
from sklearn import model_selection  # used to split the dataset into training and testing data
from sklearn.metrics import classification_report  # to print the classification report after testing model
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier  # for the classification model options


# This function parses the command line argument to decide which variety of Random Forest classifier to use
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Random Forest Classifier')
    parser.add_argument('--classifier', dest='classifier_type', 
                        required=True, choices=['randomforest', 'extremerandomforest'],
                        help="Type of classifier to use; --classifier can be either "
                        "'randomforest' or 'extremerandomforest'")
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type

    # Load the dataset, expected to be a .csv with a header row
    input_file = './phpV5QYya.csv'
    data = np.loadtxt(input_file, delimiter=',', skiprows=1)
    columns, classes = data[:, :-1], data[:, -1]

    # Split the dataset into data for training and testing
    columns_train, columns_test, classes_train, classes_test = \
        model_selection.train_test_split(columns, classes, test_size=0.20)

    # Set classifier type based on command line argument
    params = {'n_estimators': 10, 'max_depth': 4, 'random_state': 0}
    if classifier_type == 'randomforest':
        classifier = RandomForestClassifier(**params)
    else:
        classifier = ExtraTreesClassifier(**params)

    # Build a forest of trees from the training data
    classifier.fit(columns_train, classes_train)

    # Make predictions on the testing data
    classes_test_pred = classifier.predict(columns_test)

    # Print classification report
    print("\n" + "#"*15 + " Random Forest Results " + "#"*15)
    print(classification_report(classes_test, classes_test_pred))
    print("\n" + "#"*40)

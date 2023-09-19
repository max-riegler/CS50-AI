import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename, newline="") as file:
        # Read file into reader and define empty lists and a conversion dictionary for the months, visitor type, weekend, and TrueFalse
        reader = csv.DictReader(file, delimiter=",")
        evidence = []
        labels = []
        months = {"Jan" : 0, "Feb" : 1, "Mar" : 2, "Apr" : 3, "May" : 4, "June" : 5, "Jul" : 6, "Aug" : 7, "Sep" : 8, "Oct" : 9, "Nov" : 10, "Dec" : 12}
        visitors = {"Other" : 0, "New_Visitor" : 0, "Returning_Visitor" : 1}
        TrueFalse = {"FALSE" : 0, "TRUE" : 1}
        # Loop through every row in reader
        for row in reader:
            
            evidence_helper = []

            # Append Evidence to List of Lists
            evidence_helper.append(int(row["Administrative"]))
            evidence_helper.append(float(row["Administrative_Duration"]))
            evidence_helper.append(int(row["Informational"]))
            evidence_helper.append(float(row["Informational_Duration"]))
            evidence_helper.append(int(row["ProductRelated"]))
            evidence_helper.append(float(row["ProductRelated_Duration"]))
            evidence_helper.append(float(row["BounceRates"]))
            evidence_helper.append(float(row["ExitRates"]))
            evidence_helper.append(float(row["PageValues"]))
            evidence_helper.append(float(row["SpecialDay"]))
            evidence_helper.append(int(months[row["Month"]]))
            evidence_helper.append(int(row["OperatingSystems"]))
            evidence_helper.append(int(row["Browser"]))
            evidence_helper.append(int(row["Region"]))
            evidence_helper.append(int(row["TrafficType"]))
            evidence_helper.append(int(visitors[row["VisitorType"]]))
            evidence_helper.append(int(TrueFalse[row["Weekend"]]))

            # Add evidence line to evidence
            evidence.append(evidence_helper)

            # Append Labels to List
            labels.append(int(TrueFalse[row["Revenue"]]))
    
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    specificity = 0
    # Compare true positives and true negatives
    for i in range(len(labels)):
        if predictions[i] == 1 and labels[i] == 1:
            sensitivity += 1
        elif predictions[i] == 0 and labels[i] == 0:
            specificity += 1
    # Normalize to 1
    sensitivity /= labels.count(1)
    specificity /= labels.count(0)

    return sensitivity, specificity


if __name__ == "__main__":
    main()

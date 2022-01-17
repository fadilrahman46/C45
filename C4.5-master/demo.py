import C45
import csv


def main():
    # train a c4.5 decision tree and save the tree as an XML file
    reader = csv.reader(open('./data/dataset.csv'))
    training_obs = []
    training_cat = []
    for line in reader:
        training_obs.append(line[:-1])
        training_cat.append(line[-1])
    C45.train(training_obs, training_cat, "DecisionTree.xml")

    # test the C4.5 decision tree
    reader = csv.reader(open('./data/datatest.csv'))
    answer = []
    testing_obs = []
    for line in reader:
        testing_obs.append(line[:-1])
        answer.append(line[-1])
    answer.pop(0)

    prediction = C45.predict("DecisionTree.xml", testing_obs)
    err = 0
    for i in range(len(answer)):
        if not answer[i] == prediction[i]:
            err += 1
    print("error rate=", round(float(err) / len(prediction)*100, 2), "%")

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
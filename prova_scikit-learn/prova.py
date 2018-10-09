from sklearn import tree
from random import randint, shuffle

data_set_name = "tic-tac-toe"  # krkopt, poker-hand-testing, poker-hand-training-true, tic-tac-toe
data_root = "../data_set/" + data_set_name + ".data"
test_set_relation = 1 / 3
cross_validation_number = 3

file = open(data_root, "r")
data = file.readlines()
file.close()

print("\nElementi del data set: " + str(len(data)) + "\n")

if data_set_name == "tic-tac-toe":
    for i in range(0, len(data)):
        data[i] = data[i].replace("b", "0")
        data[i] = data[i].replace("x", "1")
        data[i] = data[i].replace("o", "2")

elif data_set_name == "krkopt":
    for i in range(0, len(data)):
        data[i] = data[i].replace("a", "1")
        data[i] = data[i].replace("b", "2")
        data[i] = data[i].replace("c", "3")
        data[i] = data[i].replace("d", "4")
        data[i] = data[i].replace("e", "5")
        data[i] = data[i].replace("f", "6")
        data[i] = data[i].replace("g", "7")
        data[i] = data[i].replace("h", "8")

elif data_set_name == "poker-hand-training-true":
    new_data = []
    for i in data:
        d_elements = i.split(",")
        label = d_elements.pop(len(d_elements) - 1)
        d_elements = [int(j) for j in d_elements]
        q = 1
        while q < len(d_elements) - 2:
            w = q + 2
            while w < len(d_elements):
                if d_elements[q] > d_elements[w] or (d_elements[q] == d_elements[w] and d_elements[q - 1] > d_elements[w - 1]):
                    temp = d_elements[q]
                    d_elements[q] = d_elements[w]
                    d_elements[w] = temp
                    temp = d_elements[q - 1]
                    d_elements[q - 1] = d_elements[w - 1]
                    d_elements[w - 1] = temp
                w = w+2
            q = q + 2
        string = str(d_elements[0])
        for e in range(1, len(d_elements)):
            string = string + "," + str(d_elements[e])
        string = string + "," + label
        new_data.append(string)
    data = new_data.copy()


training_set = []
test_set = []


def resubstitution_method():
    global training_set, test_set

    training_set = data.copy()
    test_set = data.copy()

    resolution()


def test_set_method():
    global training_set, test_set

    training_set = data.copy()
    test_set = []

    num_of_test = len(training_set) * test_set_relation
    while len(test_set) < num_of_test:
        n = randint(0, len(training_set) - 1)
        test_set.append(training_set[n])
        training_set.pop(n)

    resolution()


def cross_validation_method():
    global training_set, test_set, i

    training_set = data.copy()
    test_set = []

    group_of_test_set = []
    for i in range(0, cross_validation_number):
        group_of_test_set.append([])
    shuffle(training_set)
    for j in range(0, len(training_set)):
        group_of_test_set[j % cross_validation_number].append(training_set[j])

    results = []
    for j in group_of_test_set:
        test_set = j
        training_set = data.copy()
        for ts in test_set:
            training_set.remove(ts)
        results.append(float(resolution()))

    print("Cross validation evaluation: " + str(int(sum(results) / len(results) * 100)) + "%\n")


def resolution():
    global i

    print("Training_set len: " + str(len(training_set)))
    print("Test_set len: " + str(len(test_set)))

    training_X = []
    training_Y = []
    test_X = []
    test_Y = []

    for i in training_set:
        elements = i.split(",")
        training_Y.append(elements[len(elements) - 1])
        elements.pop(len(elements) - 1)
        elements = [int(j) for j in elements]
        training_X.append(elements.copy())

    for i in test_set:
        elements = i.split(",")
        test_Y.append(elements[len(elements) - 1])
        elements.pop(len(elements) - 1)
        elements = [int(j) for j in elements]
        test_X.append(elements.copy())

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(training_X, training_Y)
    ris = clf.predict(test_X)

    correct_evaluation = 0
    for i in range(0, len(ris)):
        if str(ris[i]) == str(test_Y[i]):
            correct_evaluation = correct_evaluation + 1
    print("Correct evaluation: " + str(correct_evaluation) + " / " + str(len(test_Y)) +
          " = " + str(int(int(correct_evaluation) / len(test_Y) * 100)) + "%\n")

    return int(correct_evaluation) / len(test_Y)


print("\nMetodo di valutazione dell'errore: risostituzione")
resubstitution_method()

print("\nMetodo di valutazione dell'errore: test set")
test_set_method()

print("\nMetodo di valutazione dell'errore: cross validation")
cross_validation_method()

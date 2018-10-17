import graphviz
import prova_albero.prova as our_decision_tree_classifier
from sklearn import tree
from random import randint, shuffle
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

data_set_name = "tic-tac-toe"  # krkopt, poker-hand-testing, poker-hand-training-true, tic-tac-toe
data_root = "../data_set/" + data_set_name + ".data"
test_set_relation = 3 / 10
cross_validation_number = 5  # di solito 5 o 10
pre_pruning_minimum_n_object = 1
pre_pruning_no_useless_split = True
our_tree = True
scikit_tree = True
print_tree = True

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
                if d_elements[q] > d_elements[w] or (
                        d_elements[q] == d_elements[w] and d_elements[q - 1] > d_elements[w - 1]):
                    temp = d_elements[q]
                    d_elements[q] = d_elements[w]
                    d_elements[w] = temp
                    temp = d_elements[q - 1]
                    d_elements[q - 1] = d_elements[w - 1]
                    d_elements[w - 1] = temp
                w = w + 2
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

    resolution("resubstitution")


def test_set_method():
    global training_set, test_set

    training_set = data.copy()
    test_set = []

    num_of_test = len(training_set) * test_set_relation
    while len(test_set) < num_of_test:
        n = randint(0, len(training_set) - 1)
        test_set.append(training_set[n])
        training_set.pop(n)

    resolution("test_set")


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

    scikit_results = []
    our_results = []
    i = 1
    for j in group_of_test_set:
        test_set = j
        training_set = data.copy()
        for ts in test_set:
            training_set.remove(ts)

        if scikit_tree and our_tree:
            s_r, o_r = resolution("cross_validation_" + str(i))
            scikit_results.append(float(s_r))
            our_results.append(float(o_r))
        elif scikit_tree:
            s_r = resolution("cross_validation_" + str(i))
            scikit_results.append(float(s_r))
        elif our_tree:
            o_r = resolution("cross_validation_" + str(i))
            our_results.append(float(o_r))
        i += 1

    if scikit_tree:
        print("Scikit cross validation evaluation: " + str(int(sum(scikit_results) / len(scikit_results) * 100)) + "%")
        if not our_tree:
            print("")
    if our_tree:
        print("Our cross validation evaluation: " + str(int(sum(our_results) / len(our_results) * 100)) + "%\n")


def print_tree(decision_tree, name):
    if print_tree:
        tree_string, a = decision_tree.export_graphviz()
        graph = graphviz.Source(tree_string)
        graph.render(name)


def print_precision(ris, test_y, text):
    correct_evaluation = 0
    for k in range(0, len(ris)):
        if str(ris[k]) == str(test_y[k]):
            correct_evaluation = correct_evaluation + 1
    print(text + str(correct_evaluation) + " / " + str(len(test_y)) + " = " +
          str(int(int(correct_evaluation) / len(test_y) * 100)) + "%")
    return correct_evaluation


def resolution(evaluation_name):
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

    scikit_correct_evaluation = 0
    our_correct_evaluation = 0

    if scikit_tree:
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(training_X, training_Y)
        Scikit_ris = clf.predict(test_X)
        if print_tree:
            dot_data = tree.export_graphviz(clf, rounded=True, special_characters=True)
            s = dot_data.split("<br/>value = [")
            for k in range(1, len(s)):
                while s[k][0] != ']':
                    s[k] = s[k][1:]
                s[0] += s[k][1:]
            dot_data = s[0]
            graph = graphviz.Source(dot_data)
            graph.render("Scikit_tree_" + evaluation_name)
        scikit_correct_evaluation = print_precision(Scikit_ris, test_Y, "Scikit correct evaluation: ")

    if our_tree:
        clf = our_decision_tree_classifier
        decision_tree = clf.fit(training_X, training_Y, clf.gini(training_Y),
                                pre_pruning_no_useless_split=pre_pruning_no_useless_split,
                                pre_pruning_minimum_n_object=pre_pruning_minimum_n_object)
        Our_ris = clf.predict(test_X, decision_tree)
        print_tree(decision_tree, "Our_tree_" + evaluation_name)
        our_correct_evaluation = print_precision(Our_ris, test_Y, "Our correct evaluation: ")

        """
        decision_tree_pruning = decision_tree
        our_correct_evaluation = 0  # TODO da togliere questa riga
        clf.post_pruning(decision_tree_pruning, 0.4)
        Our_ris = clf.predict(test_X, decision_tree_pruning)
        if print_tree:
            tree_string, a = decision_tree_pruning.export_graphviz()
            graph = graphviz.Source(tree_string)
            graph.render("Our_tree_with_post_pruning" + evaluation_name)
        for i in range(0, len(Our_ris)):
            if str(Our_ris[i]) == str(test_Y[i]):
                our_correct_evaluation = our_correct_evaluation + 1
        print("Our correct evaluation with post pruning: " + str(our_correct_evaluation) + " / " + str(len(test_Y)) +
              " = " + str(int(int(our_correct_evaluation) / len(test_Y) * 100)) + "%")
        """
    print("")
    if our_tree and scikit_tree:
        return int(scikit_correct_evaluation) / len(test_Y), int(our_correct_evaluation) / len(test_Y)
    if scikit_tree:
        return int(scikit_correct_evaluation) / len(test_Y)
    if our_tree:
        return int(our_correct_evaluation) / len(test_Y)


# print("\nMetodo di valutazione dell'errore: risostituzione")
# resubstitution_method()

print("\nMetodo di valutazione dell'errore: test set")
test_set_method()

# print("\nMetodo di valutazione dell'errore: cross validation")
# cross_validation_method()

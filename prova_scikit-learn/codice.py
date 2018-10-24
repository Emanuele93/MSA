import graphviz
import prova_albero.prova as our_decision_tree_classifier
from sklearn import tree
from random import randint, shuffle
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

data_set_name = "poker-hand-training-true"  # krkopt, poker-hand-testing, poker-hand-training-true, tic-tac-toe
data_root = "../data_set/" + data_set_name + ".data"
our_tree = True
scikit_tree = True
print_tree = False
best_scikit_correct_evaluation = 0
best_our_correct_evaluation = 0
best_our_correct_evaluation_number = -1
count = 0

file = open(data_root, "r")
data = file.readlines()
file.close()
for the_file in os.listdir("trees"):
    os.remove('trees/' + the_file)

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
    global training_set, test_set, best_our_correct_evaluation, best_scikit_correct_evaluation, test_set_relation, \
        pre_pruning_no_useless_split, pre_pruning_minimum_n_object, post_pruning_pessimistic, post_pruning_reduced_error

    test_set_relation = 0
    pre_pruning_no_useless_split = True
    pre_pruning_minimum_n_object = range(1, 2)  # secondo numero escluso (15)
    post_pruning_pessimistic = range(0, 1)  # secondo numero escluso (da 2 a 10) -> il valore verrà diviso per 10
    post_pruning_reduced_error = False

    best_our_correct_evaluation = 0
    best_scikit_correct_evaluation = 0

    training_set = data.copy()
    test_set = data.copy()

    shuffle(training_set)
    resolution("resubstitution")


def test_set_method():
    global training_set, test_set, best_our_correct_evaluation, best_scikit_correct_evaluation, test_set_relation, \
        pre_pruning_no_useless_split, pre_pruning_minimum_n_object, post_pruning_pessimistic, post_pruning_reduced_error

    test_set_relation = 3 / 10
    pre_pruning_no_useless_split = True
    pre_pruning_minimum_n_object = range(1, 2)  # secondo numero escluso (15)
    post_pruning_pessimistic = range(0, 1)  # secondo numero escluso (da 2 a 10) -> il valore verrà diviso per 10
    post_pruning_reduced_error = False

    best_our_correct_evaluation = 0
    best_scikit_correct_evaluation = 0

    training_set = data.copy()
    test_set = []

    shuffle(training_set)
    num_of_test = len(training_set) * test_set_relation
    while len(test_set) < num_of_test:
        n = randint(0, len(training_set) - 1)
        test_set.append(training_set[n])
        training_set.pop(n)

    resolution("test_set")


def cross_validation_method():
    global training_set, test_set, best_our_correct_evaluation, best_scikit_correct_evaluation, test_set_relation, \
        pre_pruning_no_useless_split, pre_pruning_minimum_n_object, post_pruning_pessimistic, post_pruning_reduced_error

    cross_validation_number = 5  # di solito 5 o 10
    test_set_relation = 3 / 10
    pre_pruning_no_useless_split = True
    pre_pruning_minimum_n_object = range(1, 2)  # secondo numero escluso (15)
    post_pruning_pessimistic = range(0, 1)  # secondo numero escluso (da 2 a 10) -> il valore verrà diviso per 10
    post_pruning_reduced_error = False

    best_our_correct_evaluation = 0
    best_scikit_correct_evaluation = 0

    training_set = data.copy()
    test_set = []

    group_of_test_set = []
    for k in range(0, cross_validation_number):
        group_of_test_set.append([])
    shuffle(training_set)
    for j in range(0, len(training_set)):
        group_of_test_set[j % cross_validation_number].append(training_set[j])

    scikit_results = []
    our_results = []
    k = 0
    for j in group_of_test_set:
        test_set = j
        training_set = data.copy()
        for ts in test_set:
            training_set.remove(ts)
        k += 1
        print("Cross validation, group " + str(k))
        if scikit_tree and our_tree:
            s_r, o_r = resolution("cross_validation_" + str(k))
            scikit_results.append(float(s_r))
            our_results.append(float(o_r))
        elif scikit_tree:
            s_r = resolution("cross_validation_" + str(k))
            scikit_results.append(float(s_r))
        elif our_tree:
            o_r = resolution("cross_validation_" + str(k))
            our_results.append(float(o_r))

    if scikit_tree:
        print("Scikit cross validation evaluation: " + str(int(sum(scikit_results) / len(scikit_results) * 100)) + "%")
        if not our_tree:
            print("")
    if our_tree:
        print("Our cross validation evaluation: " + str(int(sum(our_results) / len(our_results) * 100)) + "%\n")


def draw_tree(decision_tree, name):
    global count
    if print_tree:
        tree_string, a = decision_tree.export_graphviz()
        graph = graphviz.Source(tree_string)
        count += 1
        graph.render("trees/" + str(count) + name)
        os.remove("trees/" + str(count) + name)


def print_precision(ris, test_y, text, num=0):
    correct_evaluation = 0
    num_str = "%"
    if num > 0:
        num_str = "%   (" + str(num) + " nodes)"
    for k in range(0, len(ris)):
        if str(ris[k]) == str(test_y[k]):
            correct_evaluation = correct_evaluation + 1
    print(text + str(correct_evaluation) + " / " + str(len(test_y)) + " = " +
          str(int(int(correct_evaluation) / len(test_y) * 100)) + num_str)
    return correct_evaluation


def resolution(evaluation_name):
    global i, best_scikit_correct_evaluation, best_our_correct_evaluation, best_our_correct_evaluation_number
    best_our_correct_evaluation_number = -1

    training_X = []
    training_Y = []
    test_X = []
    test_Y = []
    validation_set_X = []
    validation_set_Y = []

    print("Training_set len: " + str(len(training_set)))
    print("Test_set len: " + str(len(test_set)))

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
    local_best_our_evaluation = 0

    if scikit_tree:
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(training_X, training_Y)
        Scikit_ris = clf.predict(test_X)
        scikit_correct_evaluation = print_precision(Scikit_ris, test_Y, "Scikit correct evaluation: ")
        if best_scikit_correct_evaluation <= scikit_correct_evaluation:
            best_scikit_correct_evaluation = scikit_correct_evaluation
            if print_tree:
                dot_data = tree.export_graphviz(clf, rounded=True, special_characters=True)
                s = dot_data.split("<br/>value = [")
                for k in range(1, len(s)):
                    while s[k][0] != ']':
                        s[k] = s[k][1:]
                    s[0] += s[k][1:]
                dot_data = s[0]
                graph = graphviz.Source(dot_data)
                graph.render("trees/Scikit_tree_" + evaluation_name)
                os.remove("trees/Scikit_tree_" + evaluation_name)

    if our_tree:
        clf = our_decision_tree_classifier
        if post_pruning_reduced_error:
            decision_tree, num = clf.fit(training_X, training_Y, clf.gini(training_Y),
                                         pre_pruning_no_useless_split=pre_pruning_no_useless_split,
                                         pre_pruning_minimum_n_object=1)
            Our_ris = clf.predict(test_X, decision_tree)
            our_correct_evaluation = print_precision(Our_ris, test_Y, "Our correct evaluation: ", num=num)

            if best_our_correct_evaluation < our_correct_evaluation \
                    or ((best_our_correct_evaluation_number > num or best_our_correct_evaluation_number == -1)
                        and best_our_correct_evaluation <= our_correct_evaluation):
                best_our_correct_evaluation = our_correct_evaluation
                best_our_correct_evaluation_number = num
                draw_tree(decision_tree, " (" + str(our_correct_evaluation) + "-" + str(len(test_Y)) + " "
                          + str(num) + "n) Our_tree_" + evaluation_name)
            if local_best_our_evaluation < our_correct_evaluation:
                local_best_our_evaluation = our_correct_evaluation

            num_of_validation = len(data) * 1 / 10
            while len(validation_set_X) < num_of_validation:
                validation_set_X.append(training_X[0])
                training_X.pop(0)
                validation_set_Y.append(training_Y[0])
                training_Y.pop(0)
            print("Training_set_Growing len: " + str(len(training_X)))
            print("Training_set_Validation len: " + str(len(validation_set_X)))

        for i in pre_pruning_minimum_n_object:
            decision_tree, num = clf.fit(training_X, training_Y, clf.gini(training_Y),
                                         pre_pruning_no_useless_split=pre_pruning_no_useless_split,
                                         pre_pruning_minimum_n_object=i)
            for j in post_pruning_pessimistic:
                clf = our_decision_tree_classifier
                str_name = ""
                if post_pruning_reduced_error:
                    str_name = " with_growing_set"
                if i > 1:
                    str_name += " with_pre_cut_at_" + str(i) + "_elements"
                if j > 0:
                    str_name += " with_post_cut_at_" + str(j / 10) + "_of_gain"
                    t_num, decision_tree = clf.post_pruning_pessimistic_method(decision_tree, j / 10)
                    num -= t_num
                Our_ris = clf.predict(test_X, decision_tree)
                our_correct_evaluation = \
                    print_precision(Our_ris, test_Y, "Our correct evaluation" + str_name + ": ", num=num)

                if best_our_correct_evaluation < our_correct_evaluation \
                        or ((best_our_correct_evaluation_number > num or best_our_correct_evaluation_number == -1)
                            and best_our_correct_evaluation <= our_correct_evaluation):
                    best_our_correct_evaluation = our_correct_evaluation
                    best_our_correct_evaluation_number = num
                    draw_tree(decision_tree, " (" + str(our_correct_evaluation) + "-" + str(len(test_Y)) + " "
                              + str(num) + "n) Our_tree_" + evaluation_name + str_name)
                if local_best_our_evaluation < our_correct_evaluation:
                    local_best_our_evaluation = our_correct_evaluation

                if post_pruning_reduced_error:
                    temp_decision_tree = decision_tree
                    temp_num = num
                    str_name += " with_post_reduced_error"
                    temp_decision_tree, t_num = \
                        clf.post_pruning_reduced_error_method(temp_decision_tree, validation_set_X, validation_set_Y)
                    temp_num -= t_num
                    Our_ris = clf.predict(test_X, temp_decision_tree)
                    our_correct_evaluation = \
                        print_precision(Our_ris, test_Y, "Our correct evaluation" + str_name + ": ", num=temp_num)

                    if best_our_correct_evaluation < our_correct_evaluation \
                            or ((best_our_correct_evaluation_number > temp_num
                                 or best_our_correct_evaluation_number == -1)
                                and best_our_correct_evaluation <= our_correct_evaluation):
                        best_our_correct_evaluation = our_correct_evaluation
                        best_our_correct_evaluation_number = temp_num
                        draw_tree(temp_decision_tree, " (" + str(our_correct_evaluation) + "-" + str(len(test_Y)) + " "
                                  + str(temp_num) + "n) Our_tree_" + evaluation_name + str_name)
                    if local_best_our_evaluation < our_correct_evaluation:
                        local_best_our_evaluation = our_correct_evaluation

    print("")
    if our_tree and scikit_tree:
        return int(scikit_correct_evaluation) / len(test_Y), int(local_best_our_evaluation) / len(test_Y)
    if scikit_tree:
        return int(scikit_correct_evaluation) / len(test_Y)
    if our_tree:
        return int(local_best_our_evaluation) / len(test_Y)


print("\nValutazione del training error tramite risostituzione")
resubstitution_method()

print("\nValutazione del test error tramite test set")
test_set_method()

print("\nValutazione del test error tramite cross validation")
cross_validation_method()

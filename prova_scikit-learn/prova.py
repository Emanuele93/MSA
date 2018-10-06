from sklearn import tree
from random import randint

data_set_name = "krkopt"  # krkopt, poker-hand-testing, poker-hand-training-true, tic-tac-toe
data_root = "../data_set/" + data_set_name + ".data"
test_set_relation = 1/2

file = open(data_root, "r")
data = file.readlines()
file.close()

print("\nData len: " + str(len(data)))

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

training_set = data.copy()
test_set = []

num_of_test = len(training_set) / 3
while len(test_set) < num_of_test:
    n = randint(0, len(training_set) - 1)
    test_set.append(training_set[n])
    training_set.pop(n)

print("Training_set len: " + str(len(training_set)))
print("Test_set len: " + str(len(test_set)) + "\n")

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
    #print(str(ris[i]).strip() + " --> " + str(test_Y[i]).strip())
print("\nCorrect evaluation: " + str(correct_evaluation) + " / " + str(len(test_Y)))

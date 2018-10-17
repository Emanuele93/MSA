class Tree:
    def __init__(self, col, value, label, true_branch, false_branch,
                 true_branch_number, false_branch_number, gini_value, gain):
        self.col = col
        self.value = value
        self.label = label
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.true_branch_number = true_branch_number
        self.false_branch_number = false_branch_number
        self.gini_value = round(gini_value, 2)
        self.gain = gain

    def __str__(self, level=0):
        if self.col is None:
            ret = "\t" * level + "[ True: " + str(self.true_branch_number) + ", False: " \
                  + str(self.false_branch_number) + ", Gini: " + str(self.gini_value) + " ]\n"
        else:
            ret = "\t" * level + "[ X" + str(self.col) + " <= " + str(self.value) + " --> True: " \
                  + str(self.true_branch_number) + ", False: " + str(self.false_branch_number) \
                  + ", Gini: " + str(self.gini_value) + " ]\n"
        if isinstance(self.true_branch, Tree):
            ret += self.true_branch.__str__(level + 1)
        if isinstance(self.false_branch, Tree):
            ret += self.false_branch.__str__(level + 1)
        return ret

    def export_graphviz(self, number=0):
        ret = ""
        if number == 0:
            ret += "digraph Tree {\nnode [shape=box, style=\"rounded\", color=\"black\", " \
                   "fontname=helvetica] ;\nedge [fontname=helvetica] ;" + "\n"
        if self.true_branch is None and self.false_branch is None:
            ret += str(number) + "[label=<gini = " + str(self.gini_value) + "<br/>samples = " \
                   + str(self.false_branch_number + self.true_branch_number) + ">] ;" + '\n'
        else:
            ret += str(number) + " [label=<X<SUB>" + str(self.col) + "</SUB> &le; " + str(self.value) \
                   + "<br/>gini = " + str(self.gini_value) + "<br/>samples = " \
                   + str(self.false_branch_number + self.true_branch_number) + ">] ;" + '\n'
        n_f = n = number + 1
        if self.true_branch is not None:
            stringa, n_f = self.true_branch.export_graphviz(number=n_f)
            ret += stringa + str(number) + " -> " + str(number + 1)
            if number == 0:
                ret += "[labeldistance=2.5, labelangle=45, headlabel=\"True\"]"
            ret += " ;" + '\n'
            n = n_f
        if self.false_branch is not None:
            stringa, n = self.false_branch.export_graphviz(number=n_f)
            ret += stringa + str(number) + " -> " + str(n_f)
            if number == 0:
                ret += "[labeldistance=2.5, labelangle=-45, headlabel=\"False\"]"
            ret += " ;" + '\n'
        if number == 0:
            ret += "}"
        return str(ret), n + 1


def unique_counts(labels):
    results = {}
    for row in labels:
        r = row[0]
        if r not in results:
            results[r] = 0
        results[r] += 1
    return results


def gini(labels):
    total = len(labels)
    counts = unique_counts(labels)
    imp = 0.0

    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 != k2:
                p2 = float(counts[k2]) / total
                imp += p1 * p2
    return imp


def pre_pruning(labels):
    for i in labels:
        if i != labels[0]:
            return True
    return False


def splittable(data, labels, parent_gini, pre_pruning_no_useless_split, pre_pruning_minimum_n_object):
    if len(data) > 1 and (not pre_pruning_no_useless_split or pre_pruning(labels)):
        best_col = -1
        best_value = 0
        best_gain = -1
        true_best_gini = 1
        false_best_gini = 1
        for i in range(0, len(data[0])):
            for j in data:
                t_b, f_b, t_b_l, f_b_l = split(data, labels, i, j[i])
                if len(t_b) >= pre_pruning_minimum_n_object and len(f_b) >= pre_pruning_minimum_n_object:
                    gini_split_1 = gini(t_b_l)
                    gini_split_2 = gini(f_b_l)
                    p = float(len(t_b_l)) / len(t_b_l + f_b_l)
                    gain = parent_gini - p * gini_split_1 - (1 - p) * gini_split_2
                    if gain > best_gain:
                        best_col = i
                        best_value = j[i]
                        true_best_gini = gini_split_1
                        false_best_gini = gini_split_2
                        best_gain = gain
        if best_col >= 0:
            return best_col, best_value, true_best_gini, false_best_gini, best_gain
    return -1, 0, 0, 0, 0


def major_label(labels):
    max_n_label = 0
    label = ""
    for i in labels:
        n = 0
        for j in labels:
            if i == j:
                n = n + 1
        if max_n_label < n:
            max_n_label = n
            label = i
    return label


def split(data, labels, col, value):
    true_branch = []
    false_branch = []
    true_branch_labels = []
    false_branch_labels = []
    for i in range(0, len(data)):
        if data[i][col] <= value:
            true_branch.append(data[i])
            true_branch_labels.append(labels[i])
        else:
            false_branch.append(data[i])
            false_branch_labels.append(labels[i])
    return true_branch, false_branch, true_branch_labels, false_branch_labels


def fit(data, labels, gini_value, pre_pruning_no_useless_split=True, pre_pruning_minimum_n_object=1,
        post_pruning_pessimistic = True, post_pruning_reduced_error = True):
    col, value, gini_value_true, gini_value_false, gain = \
        splittable(data, labels, gini_value, pre_pruning_no_useless_split, pre_pruning_minimum_n_object)
    if col >= 0:
        true_branch, false_branch, true_branch_labels, false_branch_labels = split(data, labels, col, value)
        false_branch = fit(false_branch, false_branch_labels, gini_value_false,
                           pre_pruning_no_useless_split=pre_pruning_no_useless_split,
                           pre_pruning_minimum_n_object=pre_pruning_minimum_n_object)
        true_branch = fit(true_branch, true_branch_labels, gini_value_true,
                           pre_pruning_no_useless_split=pre_pruning_no_useless_split,
                           pre_pruning_minimum_n_object=pre_pruning_minimum_n_object)
        return Tree(col, value, major_label(labels), true_branch, false_branch,
                    len(true_branch_labels), len(false_branch_labels), gini_value, gain)
    return Tree(None, None, major_label(labels), None, None, len(data), 0, 0, 0)


def classifier(data, node):
    if node.col is None:
        return node.label
    if data[node.col] <= node.value:
        if node.true_branch.col is not None:
            return classifier(data, node.true_branch)
        return node.true_branch.label
    if node.false_branch.col is not None:
        return classifier(data, node.false_branch)
    return node.false_branch.label


def predict(data, root):
    ris = []
    for element in data:
        ris.append(classifier(element, root))
    return ris


def post_pruning(tree, min_gain):
    if tree.true_branch.col is not None:
        post_pruning(tree.true_branch, min_gain)
    if tree.false_branch.col is not None:
        post_pruning(tree.false_branch, min_gain)

    if tree.true_branch.col is None and tree.false_branch.col is None:
        if tree.gain < min_gain:
            tree.col = None
            tree.value = None
            tree.true_branch = None
            tree.false_branch = None


"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn import tree
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


data_X = [[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.], [6., 6.], [7., 7.], [8., 8.], [9., 9.], [10., 10.], [11., 11.], [12., 12.]]
data_Y = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]

n_classes = len(unique_counts(data_Y))
plot_step = 0.02

data = {"data": [], "target": [], "target_names": [], "DESCR": "no descr", "feature_names": [], "filename": "no name"}
for el in data_X:
    data["data"].append(el)
for el in data_Y:
    data["target"].append(int(el))
for el in unique_counts(data_Y):
    data["target_names"].append(el)
for el in range(0, len(data_X[0])):
    data["feature_names"].append("X" + str(el))
data["data"] = np.array(data["data"])
data["target"] = np.array(data["target"])
data["target_names"] = np.array(data["target_names"])

combinations = []
for i in range(0, len(data_X[0]) - 1):
    for j in range(i + 1, len(data_X[0])):
        combinations.append([i, j])

for pairidx, pair in enumerate(combinations):
    # We only take the two corresponding features
    X = data["data"][:, pair]
    y = data["target"]

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    dot_data = tree.export_graphviz(clf, rounded=True, special_characters=True)

    s = dot_data.split("<br/>value = [")
    for i in range(1, len(s)):
        while s[i][0] != ']':
            s[i] = s[i][1:]
        s[0] += s[i][1:]
    dot_data = s[0]

    graph = graphviz.Source(dot_data)
    graph.render("Scikit_tree")

    # Plot the decision boundary
    n = np.sqrt(len(combinations))
    if n % 1 > 0:
        n = n + 1
    n = int(n)
    if n * (n - 1) == len(combinations):
        plt.subplot(n - 1, n, pairidx + 1)
    else:
        plt.subplot(n, n, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(data["feature_names"][pair[0]])
    plt.ylabel(data["feature_names"][pair[1]])

    # Plot the training points
    for i in range(0, n_classes):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c="w", label=data["target_names"][i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.axis("tight")
plt.show()
"""

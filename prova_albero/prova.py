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

    def __copy__(self):
        return Tree(self.col.copy, self.value.copy, self.label.copy, self.true_branch.copy, self.false_branch.copy,
                    self.true_branch_number.copy, self.false_branch_number.copy, self.gini_value.copy, self.gain.copy)

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
        tested = []
        for i in range(0, len(data[0])):
            tested.append([])
        for i in range(0, len(data[0])):
            for j in data:
                if j[i] not in tested[i]:
                    tested[i].append(j[i])
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


def fit(data, labels, gini_value, pre_pruning_no_useless_split=True, pre_pruning_minimum_n_object=1, num=0):
    col, value, gini_value_true, gini_value_false, gain = \
        splittable(data, labels, gini_value, pre_pruning_no_useless_split, pre_pruning_minimum_n_object)
    if col >= 0:
        true_branch, false_branch, true_branch_labels, false_branch_labels = split(data, labels, col, value)
        false_branch, num = fit(false_branch, false_branch_labels, gini_value_false,
                                pre_pruning_no_useless_split=pre_pruning_no_useless_split,
                                pre_pruning_minimum_n_object=pre_pruning_minimum_n_object, num=num)
        true_branch, num = fit(true_branch, true_branch_labels, gini_value_true,
                               pre_pruning_no_useless_split=pre_pruning_no_useless_split,
                               pre_pruning_minimum_n_object=pre_pruning_minimum_n_object, num=num)
        return Tree(col, value, major_label(labels), true_branch, false_branch,
                    len(true_branch_labels), len(false_branch_labels), gini_value, gain), num + 1
    return Tree(None, None, major_label(labels), None, None, len(data), 0, 0, 0), num + 1


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


def post_pruning_pessimistic_method(tree, min_gain, num=0):
    if tree.true_branch is not None and tree.true_branch.col is not None:
        num, tree.true_branch = post_pruning_pessimistic_method(tree.true_branch, min_gain, num=num)
    if tree.false_branch is not None and tree.false_branch.col is not None:
        num, tree.false_branch = post_pruning_pessimistic_method(tree.false_branch, min_gain, num=num)

    if tree.true_branch is not None and tree.false_branch is not None  \
            and tree.true_branch.col is None and tree.false_branch.col is None and tree.gain < min_gain:
        tree.col = None
        tree.value = None
        tree.true_branch = None
        tree.false_branch = None
        num += 2
    return num, tree


def useless_node(root, pruning_set_x, pruning_set_y, error):
    ris = predict(pruning_set_x, root)
    n = 0
    for i in range(0, len(ris)):
        if ris[i] != pruning_set_y[i]:
            n += 1
    if error < n:
        return False, error
    return True, n


def reduced_error(root, sub_tree, pruning_set_x, pruning_set_y, error, num=0):
    if sub_tree.true_branch is not None and sub_tree.true_branch.col is not None:
        sub_tree.true_branch, num, error = reduced_error(root, sub_tree.true_branch, pruning_set_x, pruning_set_y, error, num=num)
    if sub_tree.false_branch is not None and sub_tree.false_branch.col is not None:
        sub_tree.false_branch, num, error = reduced_error(root, sub_tree.false_branch, pruning_set_x, pruning_set_y, error, num=num)

    if sub_tree.true_branch is not None and sub_tree.false_branch is not None \
            and sub_tree.true_branch.col is None and sub_tree.false_branch.col is None:
        temp_sub_tree_col = sub_tree.col
        sub_tree.col = None
        temp_sub_tree_value = sub_tree.value
        sub_tree.value = None
        temp_sub_tree_true_branch = sub_tree.true_branch
        sub_tree.true_branch = None
        temp_sub_false_true_branch = sub_tree.false_branch
        sub_tree.false_branch = None
        num += 2
        useless, new_error_n = useless_node(root, pruning_set_x, pruning_set_y, error)
        if not useless:
            sub_tree.col = temp_sub_tree_col
            sub_tree.value = temp_sub_tree_value
            sub_tree.true_branch = temp_sub_tree_true_branch
            sub_tree.false_branch = temp_sub_false_true_branch
            num -= 2
        else:
            error = new_error_n

    return sub_tree, num, error


def post_pruning_reduced_error_method(tree, pruning_set_x, pruning_set_y):
    ris_d_t = predict(pruning_set_x, tree)
    count_error = 0
    for i in range(0, len(ris_d_t)):
        if ris_d_t[i] != pruning_set_y[i]:
            count_error += 1
    tree, num, count_error = reduced_error(tree, tree, pruning_set_x, pruning_set_y, count_error)
    return tree, num

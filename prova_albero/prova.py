class Tree:
    def __init__(self, col, value, true_branch_label, false_branch_label, true_branch=None, false_branch=None):
        self.col = col
        self.value = value
        self.true_branch_label = true_branch_label
        self.false_branch_label = false_branch_label
        self.true_branch = true_branch
        self.false_branch = false_branch


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


def splittable(data, labels):
    # check_labels = len(data) > 1
    # 4 righe sotto per non dividere i data con etichette tutte uguali, riga sopra per arrivare ai singoli nodi

    check_labels = False
    for i in labels:
        if i != labels[0]:
            check_labels = True

    if check_labels:
        best_col = 0
        best_value = 0
        best_col_gini = 1
        for i in range(0, len(data[0])):
            for j in data:
                t_b, f_b, t_b_l, f_b_l = split(data, labels, i, j[i])
                if len(t_b) != 0 and len(f_b) != 0:
                    gini_value = min(gini(t_b_l), gini(f_b_l))
                    if gini_value < best_col_gini:
                        best_col = i
                        best_value = j[i]
                        best_col_gini = gini_value
        return best_col, best_value
    return -1, 0


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
        if data[i][col] > value:
            true_branch.append(data[i])
            true_branch_labels.append(labels[i])
        else:
            false_branch.append(data[i])
            false_branch_labels.append(labels[i])
    return true_branch, false_branch, true_branch_labels, false_branch_labels


def fit(data, labels):
    col, value = splittable(data, labels)
    if col >= 0:
        true_branch, false_branch, true_branch_labels, false_branch_labels = split(data, labels, col, value)
        true_branch_label = major_label(true_branch_labels)
        false_branch_label = major_label(false_branch_labels)
        false_branch = fit(false_branch, false_branch_labels)
        true_branch = fit(true_branch, true_branch_labels)
        return Tree(col, value, true_branch_label, false_branch_label, true_branch, false_branch)
    return data


def classifier(data, node):
    if data[node.col] > node.value:
        if isinstance(node.true_branch, Tree):
            return classifier(data, node.true_branch)
        return node.true_branch_label
    if isinstance(node.false_branch, Tree):
        return classifier(data, node.false_branch)
    return node.false_branch_label


def predict(data, root):
    ris = []
    for element in data:
        ris.append(classifier(element, root))
    return ris

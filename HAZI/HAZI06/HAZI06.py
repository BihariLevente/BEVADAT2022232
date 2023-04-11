import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# from src.Node import Node


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # for leaf node
        self.value = value


class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):

        self.root = None

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(
                dataset, num_samples, num_features)
            if best_split["info_gain"] > 0:
                left_subtree = self.build_tree(
                    best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(
                    best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(
                    dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -
                        1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(
                        y, left_y, right_y, "gini")
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array(
            [row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array(
            [row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(
                parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(
                parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def calculate_leaf_value(self, Y):

        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=",
                  tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):

        if tree.value != None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


HAZI06path = os.getcwd()+"\\BEVADAT2022232\\HAZI\\HAZI06"
data = pd.read_csv(HAZI06path + "\\data\\NJ.csv")

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

classifier = DecisionTreeClassifier(min_samples_split = 100, max_depth = 10)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
current_accuracy_score = accuracy_score(Y_test, Y_pred)
current_score = pd.DataFrame({"min_samples_split": [100], "max_depth": [10], "accuracy_score": [current_accuracy_score]})
print(f"Accuracy score: {current_score}")

"""
splits = [100, 500]
depths = [10, 15, 25, 50, 80, 120, 180, 250]

for split in splits:
    bestscores = pd.DataFrame({"min_samples_split": [], "max_depth": [], "accuracy_score": []})
    scores = pd.DataFrame({"min_samples_split": [], "max_depth": [], "accuracy_score": []})
    errors = pd.DataFrame({"error_min_samples_split": [], "error_max_depth": []})
    for depth in depths:
        try:
            classifier = DecisionTreeClassifier(min_samples_split=split, max_depth=depth)
            classifier.fit(X_train, Y_train)
            Y_pred = classifier.predict(X_test)
            current_accuracy_score = accuracy_score(Y_test, Y_pred)
            current_score = pd.DataFrame({"min_samples_split": [split], "max_depth": [depth], "accuracy_score": [current_accuracy_score]})
            scores = pd.concat([scores, current_score])
            print(f"Current score: {current_score}")
            if current_accuracy_score >= 0.8:
                bestscores = pd.concat([bestscores, current_score])

        except Exception:
            current_error_params = pd.DataFrame({"error_min_samples_split": [split], "error_max_depth": [depth]})
            errors = pd.concat([errors, current_error_params])
            print(f"Error with this parameters: {current_error_params}")

    if not scores.empty:
        scores.to_csv(HAZI06path + f"\\results\\scores_split{split}_depth{depths[0]}-{depths[len(depths)-1]}.csv", index=False)
    if not errors.empty:
        errors.to_csv(HAZI06path + f"\\results\\errors_split{split}_depth{depths[0]}-{depths[len(depths)-1]}.csv", index=False)
    if not bestscores.empty:
        bestscores.to_csv(HAZI06path + f"\\results\\bestscores_split{split}_depth{depths[0]}-{depths[len(depths)-1]}.csv", index=False)
"""

""" 4. Feladat: ÖSSZEFOGLALÁS
Az összefoglalás felett látható dupla for ciklussal kerestem a megfelelő paramétereket. 
A működése röviden:
Az elején eltárolom a vizsgálni kívánt split és depth értékeket egy-egy listában. A dupla for ciklus ezeken az értékeken fut végig.
Létrehozok a legjobb eredményeknek (amik 80% felett vannak), a többi eredménynek és azoknak a paramétereknek egy DataFrame-et
amik hibára futottak. A belső for ciklusban az órai kód alapján létrehozom a fát, fel fittelem, stb... végén pedig megkapom a paraméterek
alapján elért eredményt. Ha 80% felett van, akkor a bestscores DF-be tárolom el. Ellenőrzöm hogy a paraméterek miatt
hibára fut-e a program, ha igen akkor az adott "hibás" paramétereket kimentem az errors DF-be. A belső for ciklus végén pedig kimentem
azokat a DF-eket amelyek nem üresek. Utána kezdi elölről a következő split értékkel.

Elért eredmények: result mappába található rengeteg eredmény, a paraméterekkel együtt. 

Észrevétel: A split értéke nem befolyásolja az eredményt, viszont error-t dobhat ha nincs jól beállítva.

Legnagyobb hátráltató tényező: 1 darab accuracy_score kiszámolása több (3-5) percbe tellett.

Random 10 eredmény:
Split       Depth      Accuracy
1.0        	1.0        0.777333
1.0        	3.0        0.783917
2.0        	1.0        0.777333
2.0        	2.0        0.782333
3.0        	3.0        0.783917
3.0        	4.0        0.784917
3.0        	5.0        0.788583
4.0        	3.0        0.783917
4.0        	4.0        0.784917
4.0        	5.0        0.788583

Legjobb eredmény: 100-as vágás, 10-es mélység -> 80,225%-os pontosság
"""


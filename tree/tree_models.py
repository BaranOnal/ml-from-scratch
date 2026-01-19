"""
Decision Tree and Random Forest implementations from scratch.
Written for learning purposes.
"""

import numpy as np

def entropy(y):
    class_labels , counts = np.unique(y, return_counts=True)
    ps = counts/len(y)
    return -np.sum(ps * np.log2(ps))

class Node:
    def __init__(self,left,right,threshold,feature,*,value = None):
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
        self.value = value

    def is_leaf(self):
        return self.left  is None and self.right is None



class DecisionTree:
    def __init__(self,max_depth,n_feats=None,min_samples_leaf=3):
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.min_samples_leaf = min_samples_leaf

    def fit(self,X,y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats,X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self,X,y,depth = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or  n_labels == 1 or self.min_samples_leaf > n_samples: #stop condition
            leaf_value = self._most_common_label(y)
            return Node(left=None, right=None, threshold=None, feature=None, value=leaf_value)


        feat_idx = np.random.choice(n_features,self.n_feats,replace=False)
        best_feat, best_threshold = self._best_split(X,y,feat_idx)

        left_idx, right_idx = self._split(X[:,best_feat],best_threshold)
        left_node = self._grow_tree(X[left_idx],y[left_idx],depth = depth+1)
        right_node = self._grow_tree(X[right_idx],y[right_idx],depth = depth+1)
        return Node(left_node,right_node,best_threshold,best_feat)

    def _most_common_label(self,y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def _best_split(self,X,y,feat_idxs):
        best_gain = -1
        split_idx , split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(X_column,y,threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_threshold = threshold
                    split_idx = feat_idx

        return split_idx,split_threshold

    def _information_gain(self,X_column,y,split_threshold):
        parent_entropy = entropy(y)

        left_idxs,right_idxs = self._split(X_column,split_threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l,n_r = len(left_idxs),len(right_idxs)
        e_l,e_r = entropy(y[left_idxs]),entropy(y[right_idxs])

        #parent_e -(w_l * e_l + w_r * w_l)
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _split(self,X_column,split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()

        return left_idxs,right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)



class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, n_feats=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, n_feats=self.n_feats)

            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        y_pred = []
        for i in range(X.shape[0]):
            values, counts = np.unique(tree_preds[:, i], return_counts=True)
            y_pred.append(values[np.argmax(counts)])

        return np.array(y_pred)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]



from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForest(n_trees=20,max_depth=5,n_feats=int(np.sqrt(X.shape[1])))

rf.fit(X_train, y_train)
y_pred_my = rf.predict(X_test)

print("My RF accuracy:", accuracy_score(y_test, y_pred_my))

rf_sk = RandomForestClassifier(n_estimators=20,max_depth=5,max_features="sqrt",random_state=42)

rf_sk.fit(X_train, y_train)
y_pred_sk = rf_sk.predict(X_test)

print("Sklearn RF accuracy:", accuracy_score(y_test, y_pred_sk))

import csv
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import time
data = [
    ['RollNo', 'Age', 'Income', 'Student', 'Credit Rating', 'Buys Computer'],
    [1, 'Young', 'High', 'No', 'Fair', 'No'],
    [2, 'Young', 'High', 'No', 'Excellent', 'No'],
    [3, 'Middle Aged', 'High', 'No', 'Fair', 'Yes'],
    [4, 'Senior', 'Medium', 'No', 'Fair', 'Yes'],
    [5, 'Senior', 'Low', 'Yes', 'Fair', 'Yes'],
    [6, 'Senior', 'Low', 'Yes', 'Excellent', 'No'],
    [7, 'Middle Aged', 'Low', 'Yes', 'Excellent', 'Yes'],
    [8, 'Young', 'Medium', 'No', 'Fair', 'No'],
    [9, 'Young', 'Low', 'Yes', 'Fair', 'Yes'],
    [10, 'Senior', 'Medium', 'Yes', 'Fair', 'Yes'],
    [11, 'Young', 'Medium', 'Yes', 'Excellent', 'Yes'],
    [12, 'Middle Aged', 'Medium', 'No', 'Excellent', 'Yes'],
    [13, 'Middle Aged', 'High', 'Yes', 'Fair', 'Yes'],
    [14, 'Senior', 'Medium', 'No', 'Excellent', 'No']
]
with open('student_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
print("CSV file created successfully!")
df = pd.read_csv('student_data.csv')
print(df.to_string())
df = df.drop(columns=['RollNo'])
dataset = df.values.tolist()
random.shuffle(dataset)
split = int(0.8 * len(dataset))
train_data = dataset[:split]
test_data = dataset[split:]
print("\nTraining Data:")
for row in train_data:
    print(row)
print("\nTesting Data:")
for row in test_data:
    print(row)
def class_counts(dataset):
    counts = {}
    for row in dataset:
        label = row[-1]
        if label not in counts:
            counts[label] = 1
        else:
            counts[label] += 1
    return counts
def entropy(dataset):
    counts = class_counts(dataset)
    total = len(dataset)
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)
    return entropy
def split_dataset(dataset, attribute):
    splits = {}
    for row in dataset:
        value = row[attribute]
        if value not in splits:
            splits[value] = []
        splits[value].append(row)
    return splits
def information_gain(dataset, attribute):
    parent_entropy = entropy(dataset)
    splits = split_dataset(dataset, attribute)
    total = len(dataset)
    weighted_entropy = 0.0
    for subset in splits.values():
        weight = len(subset) / total
        weighted_entropy += weight * entropy(subset)
    gain = parent_entropy - weighted_entropy
    return gain
def split_information(dataset, attribute):
    splits = split_dataset(dataset, attribute)
    total = len(dataset)
    split_info = 0.0
    for subset in splits.values():
        weight = len(subset) / total
        split_info -= weight * math.log2(weight)
    return split_info
def gain_ratio(dataset, attribute):
    gain = information_gain(dataset, attribute)
    split_info = split_information(dataset, attribute)
    if split_info == 0:
        return 0
    return gain / split_info
def gini(dataset):
    counts = class_counts(dataset)
    total = len(dataset)
    gini = 1.0
    for count in counts.values():
        probability = count / total
        gini -= probability ** 2
    return gini
def gini_index(dataset, attribute):
    splits = split_dataset(dataset, attribute)
    total = len(dataset)
    weighted_gini = 0.0
    for subset in splits.values():
        weight = len(subset) / total
        weighted_gini += weight * gini(subset)
    return weighted_gini
class Node:
    def __init__(self, attribute_index=None, label=None):
        self.attribute_index = attribute_index
        self.label = label
        self.children = {}
        self.majority_class = None
def build_tree(dataset, attributes, criterion):
    counts = class_counts(dataset)
    if len(counts) == 1:
        return Node(label=list(counts.keys())[0])
    if not attributes:
        majority_class = max(counts, key=counts.get)
        return Node(label=majority_class)
    best_attribute = criterion(dataset, attributes)
    node = Node(attribute_index=best_attribute)
    node.majority_class = max(counts, key=counts.get)
    splits = split_dataset(dataset, best_attribute)
    remaining_attributes = attributes.copy()
    remaining_attributes.remove(best_attribute)
    for value, subset in splits.items():
        child = build_tree(
            subset,
            remaining_attributes,
            criterion
        )
        node.children[value] = child
    return node
def id3_selector(dataset, attributes):
    best_gain = -1
    best_attribute = None
    for attribute in attributes:
        gain = information_gain(dataset, attribute)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    return best_attribute
def c45_selector(dataset, attributes):
    best_ratio = -1
    best_attribute = None
    for attribute in attributes:
        ratio = gain_ratio(dataset, attribute)
        if ratio > best_ratio:
            best_ratio = ratio
            best_attribute = attribute
    return best_attribute
def cart_selector(dataset, attributes):
    best_gini = float("inf")
    best_attribute = None
    for attribute in attributes:
        gini_value = gini_index(dataset, attribute)
        if gini_value < best_gini:
            best_gini = gini_value
            best_attribute = attribute
    return best_attribute
attributes = [0, 1, 2, 3]
start = time.time()
id3_tree = build_tree(
    train_data,
    attributes.copy(),
    id3_selector
)
id3_time = time.time() - start
start = time.time()
c45_tree = build_tree(
    train_data,
    attributes.copy(),
    c45_selector
)
c45_time = time.time() - start
start = time.time()
cart_tree = build_tree(
    train_data,
    attributes.copy(),
    cart_selector
)
cart_time = time.time() - start
def predict(node, sample):
    if node.label is not None:
        return node.label
    attribute = node.attribute_index
    value = sample[attribute]
    if value not in node.children:
        return node.majority_class
    child = node.children[value]
    return predict(child, sample)
id3_predictions = []
for row in test_data:
    prediction = predict(id3_tree, row)
    id3_predictions.append(prediction)
c45_predictions = []
for row in test_data:
    prediction = predict(c45_tree, row)
    c45_predictions.append(prediction)
cart_predictions = []
for row in test_data:
    prediction = predict(cart_tree, row)
    cart_predictions.append(prediction)
actual_labels = [row[-1] for row in test_data]
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / len(actual)
id3_accuracy = accuracy(actual_labels, id3_predictions)
c45_accuracy = accuracy(actual_labels, c45_predictions)
cart_accuracy = accuracy(actual_labels, cart_predictions)
def confusion_matrix(actual,predicted):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(actual)):
        if actual[i] == 'Yes' and predicted[i] == 'Yes':
            tp += 1
        elif actual[i] == 'No' and predicted[i] == 'No':
            tn += 1
        elif actual[i] == 'No' and predicted[i] == 'Yes':
            fp += 1
        elif actual[i] == 'Yes' and predicted[i] == 'No':
            fn += 1
    return tp, tn, fp, fn
def precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)   
def recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)
def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall) 
id3_tp, id3_tn, id3_fp, id3_fn = confusion_matrix(
    actual_labels,
    id3_predictions
)

c45_tp, c45_tn, c45_fp, c45_fn = confusion_matrix(
    actual_labels,
    c45_predictions
)
cart_tp, cart_tn, cart_fp, cart_fn = confusion_matrix(
    actual_labels,
    cart_predictions
)
id3_precision = precision(id3_tp, id3_fp)
id3_recall = recall(id3_tp, id3_fn)
id3_f1 = f1_score(
    id3_precision,
    id3_recall
)
c45_precision = precision(c45_tp, c45_fp)
c45_recall = recall(c45_tp, c45_fn)
c45_f1 = f1_score(
    c45_precision,
    c45_recall
)
cart_precision = precision(cart_tp, cart_fp)
cart_recall = recall(cart_tp, cart_fn)
cart_f1 = f1_score(
    cart_precision,
    cart_recall
)
def tree_depth(node):
    if node.label is not None:
        return 0
    depths = []
    for child in node.children.values():
        depths.append(tree_depth(child))
    return 1 + max(depths)
def count_leaves(node):
    if node.label is not None:
        return 1
    count = 0
    for child in node.children.values():
        count += count_leaves(child)
    return count
id3_depth = tree_depth(id3_tree)
id3_leaves = count_leaves(id3_tree)
c45_depth = tree_depth(c45_tree)
c45_leaves = count_leaves(c45_tree)
cart_depth = tree_depth(cart_tree)
cart_leaves = count_leaves(cart_tree)
results = {
    "Algorithm": [
        "ID3",
        "C4.5",
        "CART"
    ],
    "Accuracy": [
        id3_accuracy,
        c45_accuracy,
        cart_accuracy
    ],
    "Precision": [
        id3_precision,
        c45_precision,
        cart_precision
    ],
    "Recall": [
        id3_recall,
        c45_recall,
        cart_recall
    ],
    "F1 Score": [
        id3_f1,
        c45_f1,
        cart_f1
    ],
    "Tree Depth": [
        id3_depth,
        c45_depth,
        cart_depth
    ],
    "Leaf Nodes": [
        id3_leaves,
        c45_leaves,
        cart_leaves
    ],
    "Training Time": [
        id3_time,
        c45_time,
        cart_time
    ]
}
comparison = pd.DataFrame(results)
print("\nFinal Comparison:")
print(comparison.to_string(index=False))
def find_best_metric(comparison, metric, higher=True):
    if higher:
        best_index = comparison[metric].idxmax()
    else:
        best_index = comparison[metric].idxmin()
    return comparison.loc[best_index, "Algorithm"]
print("Best Accuracy:",
      find_best_metric(comparison, "Accuracy"))
print("Best Precision:",
      find_best_metric(comparison, "Precision"))
print("Best Recall:",
      find_best_metric(comparison, "Recall"))
print("Best F1 Score:",
      find_best_metric(comparison, "F1 Score"))
print("Fastest Training:",
      find_best_metric(comparison, "Training Time", higher=False))
print("Simplest Tree:",
      find_best_metric(comparison, "Tree Depth", higher=False))
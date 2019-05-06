
# coding: utf-8
from __future__ import division
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import shuffle
import datetime

# get_ipython().magic(u'matplotlib inline')
mpl.rc('figure', figsize=[12,8])  #set the default figure size
import itertools, random, math
df = pd.read_csv('./arrhythmia.csv', header=None, na_values="?")

# Replace each missing value with the mode
# The preferred pandas function for finding missing values is isnull()
for i in range(280):
    if df[i].isnull().sum() > 0:
        df.iloc[:,i].fillna(df[i].mode()[0], inplace=True)


# Uses nan_to_num twice, once to convert log2 of 0 to -inf and the second to convert that times 0 to 0.
def get_data_entropy(data):
    classes, counts = np.unique(data, return_counts=True)
    probabilities = counts /len(data)
    entropies = np.nan_to_num(np.multiply(probabilities, np.nan_to_num(np.log2(probabilities))))
    return sum(entropies) * -1


# Node class to store children of tree and print instructions.
class Node(object):
    def __init__(self, data, name, split=None):
        self.name = name
        self.node_type = None
        self.label = None
        self.data = data
        self.split = split
        self.children = []
        
    def __repr__(self):
        data = self.data
        if self.node_type != 'leaf':
            s = (f"{self.name} Internal node with {len(data)} rows; split "
                 f"{self.split.split_column} at {self.split.point:.2f} for children with" 
                 f" {[len(p) for p in self.split.partitions()]} rows"
                 f" and infomation gain {self.split.info_gain:.5f}")
        else:
            s = (f"{self.name} Leaf with {len(data)} rows, and label"
                 f" {self.label}")
        return s

class Split(object):
    def __init__(self, data, class_column, split_column, total_data_entropy=None, point=None):
        self.data = data[data[:,split_column].argsort()]
        self.class_column = class_column
        self.split_column = split_column
        self.info_gain = None
        self.point = point
        self.partition_list = None # stores the data points on each side of the split
        self.total_data_entropy = total_data_entropy
        self.compute_info_gain()

    def compute_info_gain(self):
        last_split_tested = None
        last_class = None
        max_entropy = 0
        row_with_max_entropy = -1

        for row in range(len(self.data)):
            if not self.data[row, self.split_column] == last_split_tested and not self.data[row, self.class_column] == last_class:
                entropy_score = self.get_entropy(row)
                last_split_tested = self.data[row, self.split_column]
                last_class = self.data[row, self.class_column]
                if entropy_score > max_entropy:
                    max_entropy = entropy_score
                    row_with_max_entropy = row
        self.info_gain = max_entropy
        if row_with_max_entropy > -1:
            self.point = np.mean(self.data[row_with_max_entropy - 1 : row_with_max_entropy + 1, self.split_column], )

        
    def partitions(self):
        '''Get the two partitions (child nodes) for this split.'''
        if self.partition_list:
            # This check ensures that the list is computed at most once.  Once computed 
            # it is stored
            return self.partition_list
        data = self.data
        split_column = self.split_column
        lt_partition = data[data[:, split_column] <= self.point]
        gt_partition = data[data[:, split_column] > self.point]
        partition_list = []

        # Skip empty partitions.
        if len(lt_partition) > 0:
            partition_list.append(lt_partition)
        if len(gt_partition) > 0:
            partition_list.append(gt_partition)

        self.partition_list = partition_list
        return partition_list    
    
    # This evaluates equality based on info gain. If the other split is null, just return true.
    def __gt__(self, other):
        if other is None:
            return True
        if not isinstance(other, Split):
            raise TypeError("Split not found.")
        return self.info_gain > other.info_gain

    def get_entropy(self, split_row):

        total_entropy = 0
        data_entropy = self.total_data_entropy
        first_half = self.data[: split_row, self.class_column]
        second_half = self.data[split_row :, self.class_column]

        # Add weighted entropy values to subtract from the data's entropy
        for cut in (first_half, second_half):
            total_entropy += len(cut) / len(self.data) * get_data_entropy(cut)
        return data_entropy - total_entropy

# This creates a tree to be used as a model for predictions.
class DecisionTree(object):

    def __init__(self, max_depth=None):
        if max_depth is not None and (max_depth != int(max_depth) or max_depth < 0):
            raise Exception("Invalid max depth value.")
        self.max_depth = max_depth
        

    def fit(self, data, class_column):
        '''Fit a tree on data, in which class_column is the target.'''
        if not(class_column == -1 or class_column == len(data[0])):
            raise Exception("Invalid input")

        self.data = data
        self.class_column = class_column
        self.non_class_columns = range(len(data[0]) - 1)
        self.root = self.recursive_build_tree(self.data, depth=0, name='0')
            
    def recursive_build_tree(self, data, depth, name):
        if depth != self.max_depth - 1:
            next_split = None
            total_data_entropy = get_data_entropy(data[:, self.class_column])

            # Look for the best split among all the columns
            for split_column in self.non_class_columns:
                possible_split = Split(data, self.class_column, split_column, total_data_entropy=total_data_entropy)
                if possible_split > next_split:
                    next_split = possible_split

            new_name_number = 0
            node = Node(data, name, next_split)

            #Label it as a leaf if there was no info gain.
            if node.split.info_gain == 0 or len(next_split.partitions()) == 1:
                return self.get_leaf(data, name)

            # Add the children from the partition to node.children
            for partition in next_split.partitions():
                if depth + 1 == self.max_depth - 1:
                    child = self.get_leaf(partition,  name + '.' + str(new_name_number))
                else:
                    child = self.recursive_build_tree(partition, depth + 1, name + '.' + str(new_name_number))
                if child is not None:
                    node.children.append(child)
                new_name_number += 1

            return node
    
    def get_leaf(self, data, name):
        node = Node(data, name)
        node.node_type = 'leaf'
        node.label = stats.mode(node.data[:, -1])[0][0]
        return node
    
    def print(self):
        self.recursive_print(self.root)
    
    def recursive_print(self, node):
        print(node)
        for u in node.children:
            self.recursive_print(u)

    def predict(self, test):
        '''Runs test data through model to return prediction score of % correctly categorized. '''

        self.test_data = test
        predictions = np.zeros((len(self.test_data), 1))
        self.test_data = np.hstack((self.test_data, predictions))
        mask = np.array(np.ones((1, self.test_data.shape[0]), dtype=bool))
        self.predict_loop(self.root, mask)
        return self.get_prediction_score()

    def predict_loop(self, model_node, mask):
        if model_node.node_type == 'leaf':
            self.test_data[mask[0] == 1, -1] = model_node.label

        elif self.test_data is not None:

            left = np.array(self.test_data[:, model_node.split.split_column] <= model_node.split.point) & mask
            right = np.array(self.test_data[:, model_node.split.split_column] > model_node.split.point) & mask
            self.predict_loop(model_node.children[0], left)
            self.predict_loop(model_node.children[1], right)

    def get_prediction_score(self):
        correct_predictions = self.test_data[:, -2] == self.test_data[:, -1]
        return sum(correct_predictions) / len(correct_predictions)


def get_accuracies(training, test):
    if isinstance(training, pd.DataFrame) and len(training) > 0 and isinstance(test, pd.DataFrame) and len(test) > 0:
        training_values = training.values
        test_values = test.values
        training_accuracy = []
        test_accuracy = []
        for depth in range(2, 22, 2):
            tree = DecisionTree(depth)
            tree.fit(training_values, -1)
            training_accuracy.append(tree.predict(training_values))
            test_accuracy.append(tree.predict(test_values))
    return training_accuracy, test_accuracy

def get_averages(averages1, averages2, averages3):
    average_of_averages = []
    if len(averages1) == len(averages2) == len(averages3):
        for avg1, avg2, avg3 in zip(averages1, averages2, averages3):
            average_of_averages.append(np.mean([avg1, avg2, avg3]))
    return average_of_averages

# Create Pdf of Results and save to local folder as "Results.pdf"
def export_to_plot(training_accuracy_avgs, test_accuracy_avgs):
    depth = [x for x in range(2, 22, 2)]
    plt.plot(depth, training_accuracy_avgs, 'rs', label='Training')
    plt.plot(depth, test_accuracy_avgs, 'bs', label='Test')
    plt.title("Accuracies of Averages of Training and Test Data", fontsize=20)
    plt.legend(fontsize=20)
    plt.xlabel('Decision Tree Depth', fontsize=20)
    plt.ylabel('Pct. Accuracy', fontsize=20)
    plt.xticks(depth)
    plt.savefig("validation.pdf")


def get_samples():
     # Randomize the data
    sampledf = shuffle(df)

    # Find Cut Points
    first_cut = int(round(len(sampledf)/3, 0))
    second_cut = int(round(len(sampledf)*2/3, 0))

    # Cut the data
    sample1 = sampledf.iloc[: first_cut, :]
    sample2 = sampledf.iloc[first_cut: second_cut, :]
    sample3 = sampledf.iloc[second_cut:, :]

    # Create training data
    fold1_training = sample1.append(sample2)
    fold2_training = sample2.append(sample3)
    fold3_training = sample1.append(sample3)

    training_test_pairs = {
        'Train 1': sample1,
        'Train 2': sample2,
        'Train 3' : sample3,
        'Test 1': fold1_training,
        'Test 2' : fold2_training,
        'Test 3' : fold3_training
    }
    return training_test_pairs

def validation_curve():
    samples = get_samples()

    # Get accuracies
    fold1_training_accuracies, fold1_test_accuracies = get_accuracies(samples['Train 1'], samples['Test 1'])
    fold2_training_accuracies, fold2_test_accuracies = get_accuracies(samples['Train 2'], samples['Test 2'])
    fold3_training_accuracies, fold3_test_accuracies = get_accuracies(samples['Train 3'], samples['Test 3'])

    #average the accuracies
    training_accuracy_avgs = get_averages(fold1_training_accuracies, fold2_training_accuracies, fold3_training_accuracies)
    test_accuracy_avgs = get_averages(fold1_test_accuracies, fold2_test_accuracies, fold3_test_accuracies)

    # Plot the accuracies
    export_to_plot(training_accuracy_avgs, test_accuracy_avgs)

if __name__ == "__main__":
    start = datetime.datetime.today()
    print(start)
    validation_curve()
    print(datetime.datetime.today() - start)
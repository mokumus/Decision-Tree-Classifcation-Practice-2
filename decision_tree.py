import numpy as np
from math import log
from enum import Enum

class Measure(Enum):
	GINI = 1
	ENTROPY = 2
	MISCLASS = 3

def remove_garbage(df):
	"""
	Removes None and infinity values from given data frame, inplace.
	:param df: pandas.DataFrame
	:return: pandas.DataFrame
	"""
	df.dropna(inplace=True)
	df = df.loc[:, [i for i in df.columns]]
	indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
	return df[indices_to_keep].astype(np.float64)

def score_splits(groups, classes, measure):
	"""
	Cost function used to evaluate splits in the database.
	:param measure: gini, entropy or misclassification rate
	:param groups: 2 groups sub groups of database
	:param classes: attribute to be predicted, i.e: White Wine[0.0], Red Wine[1.0] -> classes: [0.0, 1.0]
	:return: measured score
	"""
	# calculate total number of data points
	n_datapoints = float(sum([len(group) for group in groups]))
	gini = entropy = misclass = 0.0

	for group in groups: # sum weighted gini index for each group
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		gini_score = entropy_score = misclass_score = 0.0

		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			if p != 0:
				gini_score     += p * p
				entropy_score  += -p * log(p, 2)
				misclass_score += 1 - max(p, 1 - p)
		# weight the group score by its relative size
		gini     += (1.0 - gini_score) * (size / n_datapoints)
		entropy  += entropy_score * (size / n_datapoints)
		misclass += misclass_score * (size / n_datapoints)

	if measure == Measure.GINI:
		return gini
	elif measure == Measure.ENTROPY:
		return entropy
	elif measure == Measure.MISCLASS:
		return misclass
	else:
		raise Exception('Invalid measure method')




def construct_split(index, value, db_matrix):
	"""
	Constructs two lists from db_matrix, split is based on the given value
	:param index: index of the attribute to be compared in the row
	:param value: attribute value of the splitting point
	:param db_matrix: database matrix -> array-like
	:return: left -> list() , right -> list()
	"""
	left, right = list(), list()
	for row in db_matrix:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right


def get_split(db_matrix, measure):
	"""
	Find the best split point in given db_matrix
	:param db_matrix: database matrix -> array-like
	:param measure: gini, entropy or misclassification rate
	:return: dict('index': index of the attribute compared in the row
				  'value': attribute value of the splitting point,
				  'groups': 2 groups sub groups of database)
	"""
	class_values = list(set(row[-1] for row in db_matrix))
	int_inf = 2**10000
	split_index = int_inf
	split_value = int_inf
	split_score = int_inf
	split_groups =  None
	# For all columns except the last, last one is class tag
	for index in range(len(db_matrix[0]) - 1):
		for row in db_matrix:
			groups = construct_split(index, row[index], db_matrix)
			tmp_score = score_splits(groups, class_values, measure)
			#print('X{} < {:.3f} Score={:.2f}'.format(index+1, row[index], tmp_score))
			# Update split info if new best split is found
			if tmp_score < split_score:
				split_index = index
				split_value = row[index]
				split_score = tmp_score
				split_groups = groups

	result = {'index':split_index, 'value':split_value, 'groups':split_groups}
	return result

def construct_leaf_node(group):
	"""
	Constructs a node that represents a class(White/Red) decided by which is dominant in the given group
	:param group: left + right groups -> array-like
	:return: class tag, in this case it is binary
	"""
	outcomes = [row[-1] for row in group] # List of class tags for each row
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth, measure):
	"""
	Recursively create splits until max depth is reached
	:param node: decision node
	:param max_depth: maximum depth of the decision tree
	:param min_size: minimum size of a split group
	:param depth: depth of the node
	:param measure: gini, entropy or misclassification rate
	:return: None
	"""
	left, right = node['groups']

	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = construct_leaf_node(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = construct_leaf_node(left), construct_leaf_node(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = construct_leaf_node(left)
	else:
		node['left'] = get_split(left, measure)
		split(node['left'], max_depth, min_size, depth+1, measure)
	# process right child
	if len(right) <= min_size:
		node['right'] = construct_leaf_node(right)
	else:
		node['right'] = get_split(right, measure)
		split(node['right'], max_depth, min_size, depth+1, measure)


def build_tree(db_training, max_depth, min_size , measure):
	"""
	Build a decision tree
	:param db_training: Training data to construct the rules for the decision tree
	:param max_depth: Maximum depth of the tree
	:param min_size: Minimum split size
	:param measure: gini, entropy or misclassification rate
	:return: tree root
	"""
	root = get_split(db_training, measure)
	split(root, max_depth, min_size, 1, measure)
	return root

def print_tree(node, labels, depth=0 ):
	"""
	Print decision tree recursively
	:param labels: column labels i.e: ['fixed acidity', 'alcohol', ...]
	:param node: tree node(dict{index, left, right, value})
	:param depth: node depth
	:return:
	"""
	if isinstance(node, dict):
		print('{}[{} < {:.3f}]'.format(depth*'\t', labels[node['index']] , node['value']))
		print_tree(node['left'], labels, depth+1)
		print_tree(node['right'], labels, depth+1)
	else:
		print('{}[{}]'.format(depth*'\t', 'White' if node==0.0 else 'Red'))

def predict(node, row):
	"""
	Make a prediction for the given row on the given node(should start with root)
	:param node: tree node
	:param row: data row with attributes
	:return: class tag, i.e:  0.0 or 1.0 for binary classification
	"""
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def accuracy(db_testing, tree):
	"""
	Calculates accuracy of the given decision tree on the given test data
	:param db_testing: testing data
	:param tree: constructed tree
	:return: Success rate -> between 0.0 and 1.0
	"""
	correct_predictions = 0.0
	print("Predictions: ", end="")
	for row in db_testing:
		prediction = predict(tree, row)
		print("{:.0f}".format(prediction), end="")
		if prediction == row[-1]:
			correct_predictions += 1.0
	print()
	return correct_predictions / len(db_testing)



import numpy as np
import pandas as pd

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

def gini_index(groups, classes):
	"""
	Cost function used to evaluate splits in the database.
	:param groups: 2 groups sub groups of database
	:param classes: attribute to be predicted, i.e: White Wine[0.0], Red Wine[1.0] -> classes: [0.0, 1.0]
	:return: gini index
	"""
	# calculate total number of data points
	n_datapoints = float(sum([len(group) for group in groups]))
	gini = 0.0

	for group in groups: # sum weighted gini index for each group
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class, higher is better
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_datapoints)
	return gini

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


def get_split(db_matrix):
	"""
	Find the best split point in given db_matrix
	:param db_matrix: database matrix -> array-like
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
			gini = gini_index(groups, class_values)
			#print('X{} < {:.3f} Gini={:.2f}'.format(index+1, row[index], gini))
			# Update split info if new best split is found
			if gini < split_score:
				split_index = index
				split_value = row[index]
				split_score = gini
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
def split(node, max_depth, min_size, depth):
	"""
	Recursively create splits until max depth is reached
	:param node: decision node
	:param max_depth: maximum depth of the decision tree
	:param min_size: minimum size of a split group
	:param depth: depth of the node
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
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = construct_leaf_node(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)


def build_tree(db_training, max_depth, min_size):
	"""
	Build a decision tree
	:param db_training: Training data to construct the rules for the decision tree
	:param max_depth: Maximum depth of the tree
	:param min_size: Minimum split size
	:return: tree root
	"""
	root = get_split(db_training)
	split(root, max_depth, min_size, 1)
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
	for row in db_testing:
		prediction = predict(tree, row)
		if prediction == row[-1]:
			correct_predictions += 1.0
	return correct_predictions / len(db_testing)

def main() :
	#Set display option for data frames
	pd.set_option('display.max_columns', 11)
	pd.set_option('display.width', 200)

	#Read data and remove garbage
	df = pd.read_csv('winequalityN.csv')
	df = remove_garbage(pd.DataFrame(data=df, columns=list(df.columns.values)))
	cols = df.columns.tolist()
	cols = cols[1:] + cols[0:1]	#Move wine color column to last column
	df = df[cols]
	labels = df.columns.values

	#Extract training data, sample size n
	#df_white = df[(df['type'] == 0.0)]
	#df_red = df[(df['type'] == 1.0)]
	df_training = df.sample(n=100, random_state=1)	#Mixed sample

	tree_depth = 3
	min_split = 1
	tree = build_tree(df_training.values, max_depth=tree_depth, min_size=min_split)
	print("=" * 40)
	print_tree(tree, labels)
	print("=" * 40)
	print('Min split:   {}'.format(min_split))
	print('Tree depth:  {}'.format(tree_depth))
	print('Sample Size: {}'.format(len(df_training)))
	print('Accuracy:    {:.2f}'.format(accuracy(df.values, tree)))
	print("="*40)

if __name__ == "__main__":
	main()

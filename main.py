import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import decision_tree as dt

from warnings import filterwarnings
filterwarnings('ignore')


def criteria_plot():
	plt.figure(figsize=(6,4))
	xx = np.linspace(0,1,50)
	plt.plot(xx, [2*x*(1-x) for x in xx], label='gini')
	plt.plot(xx, [4*x*(1-x) for x in xx], label='2*gini')
	plt.plot(xx, [-x* np.log2(x) - (1-x)* np.log2(1-x) for x in xx], label='entropy')
	plt.plot(xx, [1 - max(x, 1-x) for x in xx], label='missclass')
	plt.plot(xx, [2 -2*max(x, 1 - x) for x in xx], label='2*missclass')
	plt.xlabel('p+')
	plt.ylabel('criterion')
	plt.title('Criteria of quality as a function of p+ (binary classification)')
	plt.legend()
	plt.show()

def run_tests(df, df_training, labels):
	for m in dt.Measure:
		for i in range(1, 5):
			tree_depth = i
			min_split = 1
			test_set = df.values
			measure = m
			tree = dt.build_tree(df_training.values, max_depth=tree_depth, min_size=min_split, measure=measure)
			print("=" * 40)
			dt.print_tree(tree, labels)
			print('Min split:   {}'.format(min_split))
			print('Tree depth:  {}'.format(tree_depth))
			print('Train Size:  {}'.format(len(df_training)))
			print('Test Size:   {}'.format(len(test_set)))
			print('Accuracy:    {:.4f}'.format(dt.accuracy(test_set, tree)))
			print('Measure:     {}'.format(measure))
			print("=" * 40)

def main() :
	#Set display option for data frames
	pd.set_option('display.max_columns', 11)
	pd.set_option('display.width', 200)

	#Read data and remove garbage
	df = pd.read_csv('winequalityN.csv')
	df = dt.remove_garbage(pd.DataFrame(data=df, columns=list(df.columns.values)))
	cols = df.columns.tolist()
	cols = cols[1:] + cols[0:1]	#Move wine color column to last column
	#df = df[cols]
	df = df[cols].drop(['total sulfur dioxide'], axis='columns')
	labels = df.columns.values

	#Extract training data, sample size n
	df_white = df[(df['type'] == 0.0)]
	df_red = df[(df['type'] == 1.0)]
	df_training = df.sample(n=100, random_state=1)	#Mixed sample

	#run_tests(df, df_training, labels)

	tree_depth = 2
	min_split = 1
	test_set = df.values
	measure = dt.Measure.GINI
	tree = dt.build_tree(df_training.values, max_depth=tree_depth, min_size=min_split, measure=measure)
	print("=" * 40)
	dt.print_tree(tree, labels)
	print('Min split:   {}'.format(min_split))
	print('Tree depth:  {}'.format(tree_depth))
	print('Train Size:  {}'.format(len(df_training)))
	print('Test Size:   {}'.format(len(test_set)))
	print('Accuracy:    {:.4f}'.format(dt.accuracy(test_set, tree)))
	print('Measure:     {}'.format(measure))
	print("=" * 40)

	criteria_plot()
	# https://docs.google.com/spreadsheets/d/1G9dhbNj6mnbfRPGJOdwu3R1T3r_uf9Dlf83wQaUbSnw/edit#gid=0

if __name__ == "__main__":
	main()
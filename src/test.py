from evaluate import evalb
from trees import load_trees

def test(tree_path='data/22.auto.clean', evalb_path='EVALB'):
	dev_trees = load_trees(tree_path)
	score = evalb(evalb_path, dev_trees, dev_trees)
	spec = locals()
	spec.pop('dev_trees')
	for key, val in spec.items():
		print(key, val)


test()
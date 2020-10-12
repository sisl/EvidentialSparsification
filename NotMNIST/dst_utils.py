import itertools

# from Python itertools page
def powerset(s):
	"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
	return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
'''
random.random is more faster than randint

In [31]: timeit.timeit('random.random()', setup='import random', number=1000000)
Out[31]: 0.15195741500019722

In [32]: timeit.timeit('random.randint(0, 10)', setup='import random', number=1000000)
Out[32]: 2.6519588080000176

In [33]: timeit.timeit('random.randint(1, 2)', setup='import random', number=1000000)
Out[33]: 2.568905852000171

In [34]: timeit.timeit('random.randint(0, 100)', setup='import random', number=1000000)
Out[34]: 2.4685180970000147

'''
import sys
from math import sqrt
from random import randint, random
from collections import defaultdict


G_R = 1.0
G_PSIZE = 100


def gen_random_point(psize):
    # return [(randint(0, G_R), randint(0, G_R)) for item in range(psize)]
    return [(random(), random()) for item in range(psize)]


def gen_in_circle_rate(rpoints):
    summary = defaultdict(float)
    for item in rpoints:
        if sqrt(item[0]**2 + item[1]**2) < G_R:
            summary['circle'] += 1
    return summary['circle'] / len(rpoints)


def main(psize):
    rpoints = gen_random_point(psize)
    pi = 4 * gen_in_circle_rate(rpoints)
    return pi


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(main(G_PSIZE))
    else:
        print(main(int(sys.argv[1])))

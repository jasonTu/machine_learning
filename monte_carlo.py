'''
random.random is more faster than randint
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

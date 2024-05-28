import math


def get_linear_alphas(n: int):

    return [i / n for i in range(1, n + 1)]


def get_cosine_alphas(n: int):

    return [1 - (0.5 * (1. + math.cos(math.pi * (i / n)))) for i in range(1, n + 1)]

from collections import defaultdict
from functools import lru_cache

# loading data
prices = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30, 33]
length_to_price = defaultdict(int)

for i, p in enumerate(prices): length_to_price[i+1] = p


def revenue(n, cache={}, solution={}):
    if n in cache: return cache[n], solution[n]

    candidates = [ (length_to_price[n], (n, 0)) ]

    for s in range(1, n):
        split = ( revenue(s, cache, solution)[0] + revenue(n - s, cache, solution)[0], (s, n - s))
        candidates.append(split)
        # candidates.append(revenue(s, cache) + revenue(n - s, cache))

    optimal, optimal_split = max(candidates, key=lambda x: x[0])
    cache[n] = optimal

    solution[n] = optimal_split

    return optimal, solution


SOLUTION = {}


@lru_cache(maxsize=2**10)
def revenue_func_cache(n):
    candidates = [(length_to_price[n], (n, 0))]

    for s in range(1, n):
        split = (revenue(s) + revenue(n - s), (s, n - s))
        candidates.append(split)
        # candidates.append(revenue(s, cache) + revenue(n - s, cache))

    optimal, optimal_split = max(candidates, key=lambda x: x[0])

    global SOLUTION
    SOLUTION[n] = optimal_split

    return optimal


def parse_solution(n, solution):
    left, right = solution[n]

    if left == 0 or right == 0: return [left, right]
    else:
        return parse_solution(left, solution) + parse_solution(right, solution)


r, s = revenue(118)

print(s)
print(r)

print(parse_solution(118, s))
print(parse_solution(8, s))

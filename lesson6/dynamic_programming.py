# 最长公共子序列 (Longest common subsequent）

"""
S1: ABCBDAB
S2: BDCABA

S1: ATCGGACTGCAT
S2: TACTAGGACT
"""

from functools import lru_cache


@lru_cache(maxsize=2**10)
def longest_common_sub(string1, string2):
    if not string1 or not string2: return 0

    if string1[-1] == string2[-1]:
        return 1 + longest_common_sub(string1[:-1], string2[:-1])
    else:
        return max(longest_common_sub(string1[:-1], string2), longest_common_sub(string1, string2[:-1]))


SOLUTION = {}

@lru_cache(maxsize=2**10)
def edit_distance(string1, string2):
    if not string1: return len(string2)
    elif not string2: return len(string1)

    candidates = [
        (edit_distance(string1[:-1], string2) + 1, 'DEL {}'.format(string1[-1])), # del string1[-1]
        (edit_distance(string1, string2[:-1]) + 1, 'ADD {}'.format(string2[-1]))  # add string2[-1]
    ]

    if string1[-1] == string2[-1]:
        candidates.append((edit_distance(string1[:-1], string2[:-1]), ''))
    else:
        candidates.append((edit_distance(string1[:-1], string2[:-1]) + 1,
                           'REPLACE {} => {}'.format(string1[-1], string2[-1])))  # replace string2[-1] with string1[-1]

    min_distance, operator = min(candidates, key=lambda x: x[0])

    global SOLUTION
    SOLUTION[string1, string2] = operator

    return min_distance


# dynamic programming problems

def parse_edit_solution(s1, s2, solution):
    if not s1 or not s2: return ''

    if (s1, s2) not in solution:
        raise ValueError('without solution of {} to {}'.format(s1, s2))

    operator = solution[(s1, s2)]

    if 'DEL' in operator:
        op, char = operator.split()
        print('del {}'.format(char))
        parse_edit_solution(s1[:-1], s2, solution)
    elif 'ADD' in operator:
        op, char = operator.split()
        print('add {}'.format(char))
        parse_edit_solution(s1, s2[:-1], solution)
    elif 'REPLACE' in operator:
        op, char1, _, char2 = operator.split()
        print('replace {} => {}'.format(char1, char2))
        parse_edit_solution(s1[:-1], s2[:-1], solution)
    else:
        parse_edit_solution(s1[:-1], s2[:-1], solution)


if __name__ == '__main__':
    test_1 = 'ATGCGATBCAGTGATCGAGDBGAGHHDTAGDGASGDASGD'
    test_2 = 'AASYDNHASDJHASDAWACCCATATADAGAWTATACAG'

    print(longest_common_sub(test_1, test_2))

    test_3 = 'beijing'
    test_4 = 'biejie'

    print(edit_distance(test_1, test_2))

    parse_edit_solution(test_1, test_2, SOLUTION)

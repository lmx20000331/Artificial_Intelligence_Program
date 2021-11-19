from collections import defaultdict
from icecream import ic

texts = [
    ('今天天天天有天天最大的优惠！', 1),
    ('今天没有加班的话，赶快回家！', 0),
    ('不要等到明天！', 1),
]


def train(text, ys):
    counts = defaultdict(int)
    yi_num = defaultdict(int)
    for line, yi in zip(text, ys):
        yi_num[yi] += 1
        for c in set(line):
            counts[(c, yi)] += 1

    probs = defaultdict(lambda : 1/len(set(''.join(text))))

    for c_y, t in counts.items():
        c, y = c_y
        probs[(c, y)] = counts[c_y] / yi_num[y]

    return probs, {i: (yi_num[i] / len(ys)) for i in yi_num}


def predict(query, evidence, hypothesis):
    pred = {}
    for yi in hypothesis:
        prod = 1
        for c in set(query):
            prod *= evidence[(c, yi)]

        pred[yi] = prod * hypothesis[yi]

    return pred

evidence, hypothesis = train([t for t, y in texts], [y for t, y in texts])

ic(evidence)

ic(predict('明天没有优惠！', evidence, hypothesis))

#ic| predict('明天没有优惠！', evidence, hypothesis): {0: 2.739650968466617e-05, 1: 0.0018115942028985505}

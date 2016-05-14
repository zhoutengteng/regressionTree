import random
import numpy as np

def produceData():
    data = 300
    list = []
    label1 = map(lambda value: (value, random.randint(1, data), 1), map(lambda value: random.randint(1, data), range(0, data)))
    label2 = map(lambda value: (value, random.randint(1, data), 2), map(lambda value: -1 * random.randint(1, data), range(0, data)))
    label3 = map(lambda value: (value, -1 * random.randint(1, data), 3), map(lambda value: -1 * random.randint(1, data), range(0, data)))
    label4 = map(lambda value: (value, -1 * random.randint(1, data), 4), map(lambda value: random.randint(1, data), range(0, data)))

    list.extend(label1)
    list.extend(label2)
    list.extend(label3)
    list.extend(label4)
    random.shuffle(list)
    return np.array(list)

def produceDataBin():
    data = 300
    list = []
    label1 = map(lambda value: (value, random.randint(1, data), 1), map(lambda value: random.randint(1, data), range(0, data)))
    label2 = map(lambda value: (value, random.randint(1, data), -1), map(lambda value: -1 * random.randint(1, data), range(0, data)))
    label3 = map(lambda value: (value, -1 * random.randint(1, data), -1), map(lambda value: -1 * random.randint(1, data), range(0, data)))
    label4 = map(lambda value: (value, -1 * random.randint(1, data), -1), map(lambda value: random.randint(1, data), range(0, data)))

    list.extend(label1)
    list.extend(label2)
    list.extend(label3)
    list.extend(label4)
    random.shuffle(list)
    return np.array(list)


def produceDataTest():
    data = 2000
    list = []
    label1 = map(lambda value: (value, random.randint(1, data), 1), map(lambda value: random.randint(1, data), range(0, data)))
    label2 = map(lambda value: (value, random.randint(1, data), 2), map(lambda value: -1 * random.randint(1, data), range(0, data)))
    label3 = map(lambda value: (value, -1 * random.randint(1, data), 3), map(lambda value: -1 * random.randint(1, data), range(0, data)))
    label4 = map(lambda value: (value, -1 * random.randint(1, data), 4), map(lambda value: random.randint(1, data), range(0, data)))

    list.extend(label1)
    list.extend(label2)
    list.extend(label3)
    list.extend(label4)
    random.shuffle(list)
    return np.array(list)
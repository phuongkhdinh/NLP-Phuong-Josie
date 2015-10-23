
import random
import math
_NEG_INF = float('-inf')
def log_add(values):
    """
    Unlog values, Adds the values, returning the log of the addition.
    """
    x = max(values)
    if x > _NEG_INF:
        sumDiffs = 0
        for value in values:
            sumDiffs += math.exp(value - x)
        return x + math.log(sumDiffs)
    else:
        return x
def log_add2(values):
    """
    Unlog values, Adds the values, returning the log of the addition.
    """
    x = max(values)
    if x > _NEG_INF:
        sumDiffs = 0
        for value in values:
            sumDiffs += math.exp(value)
        return math.log(sumDiffs)
    else:
        return x

def main():
    for j in range(100):
        v1 = []
        v2 = []
        s1 = 0
        s2 = 0
        for i in range(1000,2000):
            m = random.random()
            s1 += m
            v1.append(math.log(s1**(1/i)))

            n = random.random()
            s2 += n
            v2.append(math.log(s2**(1/i)))
        p = log_add(v1)
        q = log_add(v2)
        r = log_add2(v1)
        s = log_add2(v2)
        if (p > q and s1 > s2) or (p < q and s1 < s2):
            #print("yay")
            q = 5
        elif(p > q and s1 < s2) or (p < q and s1 > s2):
            print("no")

main()
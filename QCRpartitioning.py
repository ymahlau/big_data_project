""" returns tuple of lists containing result[0] = T+ and result[1] containing T- """ 
def tk(sketch:list((str,float))) -> tuple(list(str),list(str)):
    result = ([str],[str])
    for x in sketch:
        for y in sketch:
            if x[0] == y[0] and x != y :
                if x[1] * y[1] >= 0 :
                    result[0].append(x)
                else:
                    result[1].append(x)
                sketch.remove(x)
                sketch.remove(y)
    return result

"""Calculates Joinability of two given input lists of keys unique within their list"""
def calculateJoinability(KC:list(str),KQ:list(str)) -> float:
    lenKC = len(KC)
    lenJoin = 0
    for x in KQ:
        for y in KC:
            if(x==y):
                lenJoin += 1
    return lenJoin/lenKC


def implementWeighting(joinability: float, correlation_coefficient:float, alpha_j, alpha_r:float) -> float:
    pass

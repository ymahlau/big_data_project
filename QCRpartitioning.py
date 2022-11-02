
def tk(sketch:list((str,float))) -> list((str, float)):
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

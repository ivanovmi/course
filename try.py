a = [[2,2,2],[2,2,2]]
b = [[2,2,2],[2,2,2]]

try:
    for i,j in a:
        for l,m in b:
            (i!=l) and (j!=m)
except Exception:
    print "Error in dimension of the matrix. Dimensions of the matrices are not the same"

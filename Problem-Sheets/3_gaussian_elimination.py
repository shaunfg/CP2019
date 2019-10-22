"""
1. Swap Rows
2. multiply rows by constants
3. add rows to rows.
"""
import numpy as np


def gaussian_elimination(A):
    """
    Operating row by row
    :param A:
    :return:
    """
    n_rows = len(A)

    for i in range(n_rows):

        # Finds the max Row
        max_row = i
        max_val = abs(A[i][i])
        for k in range(i+1,n_rows):
            max_val_2 = abs(A[k][i])
            if max_val_2 > max_val:
                max_val = max_val_2
                max_row = k

        # start from i, as from when i = 1, the first row will be zero anyways (after always zero)
        for k in range(i,n_rows+1):
            tmp = A[max_row][k]
            A[max_row][k] = A[i][k]
            A[i][k] = tmp

        for k in range(i+1,n_rows): # just used to pick the rows after i
            # This way round as A[i][i] will never be a zero
            c = - A[k][i]/ A[i][i]
            for j in range(i,n_rows+1): #same reason as before, will already be zero.
                print(i,j,A[k])
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c* A[i][j] # j as j loops through the row

    x = [0 for x in range(n_rows)]
    for i in range(n_rows-1,-1,-1):
        # Finds value of x,y,z
        x[i] = A[i][n_rows]/A[i][i]
        for k in range(i-1,-1,-1):
            #Minuses the value of e.g. =(3 -z* constant) from each row
            value = x[i] * A[k][i]
            A[k][n_rows] =  A[k][n_rows]- value
    print(A,x)

    return A

if __name__ == "__main__":
    A = [[1,4,3,4],[1,0,5,4],[2,0,3,4]]
    
    B = gaussian_elimination(A)








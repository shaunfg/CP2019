def gauss(A):
    n = len(A)

    for i in range(0, n):
        # Search for maximum in this column
        maxEl = abs(A[i][i])
        maxRow = i
        for k in range(i+1, n):
            if abs(A[k][i]) > maxEl:
                maxEl = abs(A[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        for k in range(i, n+1):
            tmp = A[maxRow][k]
            A[maxRow][k] = A[i][k]
            A[i][k] = tmp

        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            c = -A[k][i]/A[i][i]
            for j in range(i, n+1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    # Solve equation Ax=b for an upper triangular matrix A
    x = [0 for i in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = A[i][n]/A[i][i]
        num = A[i][n]
        den = A[i][i]
        for k in range(i-1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return x

print(gauss([[1,2,3,4],[1,6,5,4],[2,12,3,4]]))
""" Commented Version

    n_rows = len(A)
    final = []
    print(A)
    for i in range(n_rows):
        row = A[i]
        max_val = abs(A[i][i])
        max_row = i
        for k in range(i+1,n_rows):
            #ignore above rows, because u want to keep the zeros
            max_val_2 = abs(A[k][i])
            row_2 = A[k]
            if max_val_2 > max_val:
                max_val = max_val_2
                max_row =k

        # Swaps to avoid division by zero error!
        for k in range(i,n_rows+1):
            tmp = A[max_row][k]
            A[max_row][k] = A[i][k]
            A[i][k] = tmp
        print("--- post swap",A)

        for k in range(i+1,n_rows):
            c = -A[k][i]/A[i][i]
            final.append(c)
            for j in range(i, n_rows+1):
                if i == j:
                    check = A[k][j]
                    A[k][j] = 0
                    print("k=",k,"i==j",A,c)
                else:
                    A[k][j] += c * A[i][j]
                    print("k=",k,"i!=j",A,c)
        print(A)
        """
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def main():
    # (A)
    # a = np.array([[7, 3, 9],
    #              [3, 10, 7],
    #              [9, 7, 15]])
    # b = np.array([[20],
    #               [16],
    #               [30]])
    # x = np.linalg.solve(a, b)
    # print(x)

    # (C)
    # a = np.array([[2, 1, 2],
    #               [1, -2, 1],
    #               [1, 2, 3],
    #               [1, 1, 1]])
    #
    # b = np.array([[6],
    #               [1],
    #               [5],
    #               [2]])
    #
    # w = np.array([[1000, 0, 0, 0],
    #               [0, 1, 0, 0],
    #               [0, 0, 1, 0],
    #               [0, 0, 0, 1]])
    #
    # a_transpose = np.transpose(a)
    # x = inv(a_transpose @ w @ a) @ a_transpose @ w @ b
    # print('x= ' + str(x) + '\n')
    # r = a@x-b
    # print('r= ' + str(r))

    # (D)
    a = np.array([[2, 1, 2],
                  [1, -2, 1],
                  [1, 2, 3],
                  [1, 1, 1]])

    a_transpose = np.transpose(a)

    b = np.array([[6],
                  [1],
                  [5],
                  [2]])

    i = np.identity(3)

    l = 0.1

    x = inv(a_transpose@ a + l * i) @ a_transpose @ b

    print(x)

if __name__ == "__main__":
    main()

import numpy as np
from numpy.linalg import inv


def Q1_A():
    print("Q1_A:\n")
    a = np.array([[7, 3, 9],
                 [3, 10, 7],
                 [9, 7, 15]])
    b = np.array([[20],
                  [16],
                  [30]])
    x = np.linalg.solve(a, b)
    print(x)


def Q1_C():
    print("\nQ1_C:\n")
    a = np.array([[2, 1, 2],
                  [1, -2, 1],
                  [1, 2, 3],
                  [1, 1, 1]])

    b = np.array([[6],
                  [1],
                  [5],
                  [2]])

    w = np.array([[1000, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

    a_transpose = np.transpose(a)
    x = inv(a_transpose @ w @ a) @ a_transpose @ w @ b
    print('x= ' + str(x) + '\n')
    r = a@x-b
    print('r= ' + str(r))


def Q1_D():
    print("\nQ1_D:\n")
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


def Q3_C():
    print("\nQ3_C:\n")
    a = np.array([[5, 6, 7, 8],
                  [1, 3, 5, 4],
                  [1, 0.5, 4, 2],
                  [3, 4, 3, 1]])

    b = np.array([[0.57, 0.56, 0.8, 1],
                  [1.5, 4, 6.7, 4.9],
                  [0.2, 0.1, 1, 0.6],
                  [11, 30, 26, 10]])

    d = np.identity(4)

    for i in range(0, 4):
        a_i = a[i]
        b_i = b[i]
        d[i][i] = 1 / (np.transpose(a_i) @ a_i) * (np.transpose(a_i) @ b_i)

    print(d)


def main():
    Q1_A()
    Q1_C()
    Q1_D()
    Q3_C()


if __name__ == "__main__":
    main()

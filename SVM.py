import numpy as np
import os

from cvxopt import matrix
from cvxopt import solvers



class SVM:

    def train(self, data, lable):  # train SVM model

        y = np.asarray(lable)  # labels
        x = np.asmatrix(data)  # data matrix
        x = x.astype(np.double)
        y = y.astype(np.double)
        Q = np.zeros((len(y), len(y)))

        # build matrix for Lagrangian dual problem
        for i in range(len(y)):
            for j in range(len(y)):
             Q[i, j] = np.dot(x[i], np.transpose(x[j])) * y[i] * y[j]

        Q = matrix(0.5 * Q, tc='d')
        P = matrix(-1 * np.ones(len(y)), tc='d')

        G = matrix(np.diag(np.ones(len(y)) * (-1)), tc='d')
        h = matrix(np.zeros(len(y)), tc='d')
        A = matrix(y, (1, len(y)), tc='d')
        b = matrix([0.0])
        solvers.options['maxiters'] = 5
        sol = solvers.qp(Q, P, G, h, A, b)  # solve Lagrangian dual problem
        lamd = sol['x']

        w = [0] * 4

        for i in range(len(y)):  # calculating vector of coefficients W
            for j in range(4):
                w[j] += x[i, j] * y[i] * lamd[i]

        # calculate hyperplane parameter b
        self.__b = y[np.argmax(lamd)] - np.dot(w, np.transpose(x[np.argmax(lamd)]))
        self.__w = w


    def predict(self, data):
        x = np.asmatrix(data)
        res = []
        for i in range(len(x)):
            res.append(np.dot(self.__w, np.transpose(x[i])) + self.__b)  # predict as w * x^T + b

        return res

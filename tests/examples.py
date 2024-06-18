import math
import numpy as np

class Quadratic_func:
    def __init__(self, q_0=0, q_1=None, q_2=None, max_iter=100, max_iter_hessian=100, hessian=False):
        self.q_0 = q_0
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian
        if q_1 is None:
            self.q_2 = np.array(q_2)
            self.q_1 = np.zeros(shape=(self.q_2.shape[0], 1))
        elif q_2 is None:
            self.q_1 = np.array(q_1).reshape(-1, 1)
            self.q_2 = np.zeros(shape=(self.q_1.shape[0], self.q_1.shape[0]))
        else:
            self.q_1 = np.array(q_1).reshape(-1, 1)
            self.q_2 = np.array(q_2)

    def func(self, x):

        if x.shape[1] == 1:
            f = x.T.dot(self.q_2).dot(x) + self.q_1.T.dot(x) + self.q_0
            g = np.add(self.q_2, self.q_2.T).dot(x) + self.q_1
            if self.hessian:
                h = np.add(self.q_2, self.q_2.T)
                return f.item(), g, h
            else:
                return f.item(), g
        else:
            return np.einsum('ij,ji->i', x.T.dot(self.q_2), x) + self.q_1.T.dot(x) + self.q_0

class Rosenbrock_func:
    def __init__(self, max_iter=10000, max_iter_hessian=100, hessian=False):
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def func(self, x):

        if x.shape[1] == 1:
            x1, x2 = x[0, 0], x[1, 0]
            f = 100*math.pow((x2-math.pow(x1, 2)), 2)+math.pow((1-x1), 2)
            df_x1 = -400*x1*(x2-math.pow(x1, 2))-2*(1-x1)
            df_x2 = 200*(x2-math.pow(x1, 2))
            df_x1_x1 = -400*x2+1200*math.pow(x1, 2)+2
            df_x2_x2 = 200
            df_x1_x2 = -400*x1
            df_x2_x1 = df_x1_x2
            g = np.array([df_x1, df_x2]).reshape(-1, 1)
            if self.hessian:
                h11 = df_x1_x1
                h12 = df_x1_x2
                h21 = df_x2_x1
                h22 = df_x2_x2
                h = np.array([[h11, h12], [h21, h22]])
                return f, g, h
            else:
                return f, g
        else:
            x1, x2 = x[0], x[1]
            return 100*(x2-x1**2)**2+(1-x1)**2

class Linear_func:
    def __init__(self, a=None, max_iter=100, max_iter_hessian=100, hessian=False):
        self.a = np.array(a).reshape(-1, 1)
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def func(self, x):
        if x.shape[1] == 1:
            f = self.a.T.dot(x).item()
            g = self.a
            if self.hessian:
                h = np.zeros((2, 2))
                return f, g, h
            else:
                return f, g
        else:
            return self.a.T.dot(x).reshape(-1)

class Exponential_func:
    def __init__(self, max_iter=100, max_iter_hessian=100, hessian=False):
        self.hessian = hessian
        self.max_iter = max_iter
        self.max_iter_hessian = max_iter_hessian

    def func(self, x=None):
        x1, x2 = x[0], x[1]
        if x.shape[1] == 1:
            f = math.exp(x1+3*x2-0.1)+math.exp(x1-3*x2-0.1)+math.exp(-x1-0.1)
            df_x1 = math.exp(x1+3*x2-0.1)+math.exp(x1-3*x2-0.1)-math.exp(-x1-0.1)
            df_x2 = 3*math.exp(x1+3*x2-0.1)-3*math.exp(x1-3*x2-0.1)
            df_x1_x1 = math.exp(x1+3*x2-0.1)+math.exp(x1-3*x2-0.1)+math.exp(-x1-0.1)
            df_x2_x2 = 9*math.exp(x1+3*x2-0.1)+9*math.exp(x1-3*x2-0.1)
            df_x1_x2 = 3*math.exp(x1+3*x2-0.1)-3*math.exp(x1-3*x2-0.1)
            df_x2_x1 = df_x1_x2
            g = np.array([df_x1, df_x2]).reshape(-1, 1)
            if self.hessian:
                h11 = df_x1_x1
                h12 = df_x1_x2
                h21 = df_x2_x1
                h22 = df_x2_x2
                h = np.array([[h11, h12], [h21, h22]])
                return f, g, h
            else:
                return f, g
        else:
            return np.exp(x1+3*x2-0.1)+np.exp(x1-3*x2-0.1)+np.exp(-x1-0.1)

# Function for log barrier function
class lp_inequalities:
    def __init__(self):
        self.value_list = []
        self.m = 4

    def constraints_values(self, x):
        x1, x2 = x[0], x[1]
        if len(x1) == 1 or len(x2) == 1:
            x1, x2 = x1.item(), x2.item()
        f_1 = -x1 - x2 + 1
        f_2 = x2 - 1
        f_3 = x1 - 2
        f_4 = -x2
        f_1_grad = np.array([-1, -1]).reshape(-1, 1)
        f_2_grad = np.array([0, 1]).reshape(-1, 1)
        f_3_grad = np.array([1, 0]).reshape(-1, 1)
        f_4_grad = np.array([0, -1]).reshape(-1, 1)
        f_1_hessian = f_2_hessian = f_3_hessian = f_4_hessian = np.zeros(shape=(2, 2))
        self.value_list = [[f_1, f_1_grad, f_1_hessian], [f_2, f_2_grad, f_2_hessian], [f_3, f_3_grad, f_3_hessian], [f_4, f_4_grad, f_4_hessian]]

class qp_inequalities:
    def __init__(self):
        self.value_list = []
        self.m = 3

    def constraints_values(self, x):
        x1, x2, x3 = x[0], x[1], x[2]
        f_1 = -x1
        f_2 = -x2
        f_3 = -x3
        f_1_grad = np.array([-1, 0, 0]).reshape(-1, 1)
        f_2_grad = np.array([0, -1, 0]).reshape(-1, 1)
        f_3_grad = np.array([0, 0, -1]).reshape(-1, 1)
        f_1_hessian = f_2_hessian = f_3_hessian = np.zeros(shape=(3, 3))
        self.value_list = [[f_1, f_1_grad, f_1_hessian], [f_2, f_2_grad, f_2_hessian], [f_3, f_3_grad, f_3_hessian]]

# List of Functions
q1 = Quadratic_func(q_2=[[1, 0], [0, 1]])
q2 = Quadratic_func(q_2=[[1, 0], [0, 100]])
q3 = Quadratic_func(q_2=np.array([[(3**0.5)/2, -0.5], [0.5, (3**0.5)/2]]).T.dot(np.array([[100, 0], [0, 1]])).dot(np.array([[(3**0.5)/2, -0.5], [0.5, (3**0.5)/2]])))
qp = Quadratic_func(q_0=1, q_1=[0, 0, 2], q_2=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], hessian=True)
f_rosenbrock = Rosenbrock_func()
f_exponential = Exponential_func()
f_linear = Linear_func(a=np.array([1, 1]))
lp = Linear_func(a=[-1, -1], hessian=True)

import numpy as np
from bayeso_benchmarks import Ackley, Rastrigin, Rosenbrock

def build(math_func, dim=60, n=50000):
    print('building', math_func)
    obj_fun = eval(math_func)(dim)
    X = obj_fun.sample_uniform(2 * n)
    Y = -obj_fun.output(X)
    indices = np.argsort(Y.squeeze())[:n]
    X_low = X[indices]
    Y_low = Y[indices]

    dic2y = np.load("npy/dic2y.npy", allow_pickle=True).item()
    dic2y[math_func] = (np.min(Y_low), -obj_fun.global_minimum)
    np.save("npy/dic2y.npy", dic2y)

    np.save("npy/" + math_func + "_X.npy", X_low)
    np.save("npy/" + math_func + "_Y.npy", Y_low)

class MathBench:
    def __init__(self, math_func='Ackley'):
        self.x = np.load("npy/" + math_func + "_X.npy", allow_pickle=True)
        self.y = np.load("npy/" + math_func + "_Y.npy", allow_pickle=True)

        self.mean_x = self.x.mean(axis=0)
        self.std_x = self.x.std(axis=0)

        self.mean_y = self.y.mean(axis=0)
        self.std_y = self.y.std(axis=0)

        self.obj_func = eval(math_func)(60)
        self.min_clip = self.obj_func.bounds.squeeze()[0]
        self.max_clip = self.obj_func.bounds.squeeze()[1]

    def normalize_x(self, x_0):
        x_n = (x_0 - self.mean_x)/(self.std_x + 1e-9)
        return x_n

    def denormalize_x(self, x_n):
        x_0 = x_n * self.std_x + self.mean_x
        return x_0

    def normalize_y(self, y_0):
        y_n = (y_0 - self.mean_y) /(self.std_y + 1e-9)
        return y_n

    def predict(self, x):
        x = np.clip(x, self.min_clip, self.max_clip)
        return -self.obj_func.output(x)

if __name__ == "__main__":
    build('Ackley')
    build('Rastrigin')
    build('Rosenbrock')
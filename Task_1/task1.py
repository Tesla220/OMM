import numpy as np
import plotly
import plotly.graph_objs as go
import matplotlib
import matplotlib.pyplot as plt

layout = go.Layout(
    scene = dict(
    xaxis = dict(
        title='x',
        gridcolor="rgb(255, 255, 255)",
        zerolinecolor="rgb(255, 255, 255)",
        showbackground=True,
        backgroundcolor="rgb(200, 200, 230)"),

    yaxis = dict(
        title='t',
        gridcolor="rgb(255, 255, 255)",
        zerolinecolor="rgb(255, 255, 255)",
        showbackground=True,
        backgroundcolor="rgb(230, 200,230)"),

    zaxis = dict(
        title='u(x, t)',
        gridcolor="rgb(255, 255, 255)",
        zerolinecolor="rgb(255, 255, 255)",
        showbackground=True,
        backgroundcolor="rgb(230, 230,200)",),),
    width=800, height=600,
    margin=dict(
        r=20, b=10,
        l=10, t=10),
    )

class TES:
    def __init__(self, X_START=0, X_END=1, T_START=0, T_END=20, N=100, M=100, error=0.0001):
        self.X_START = X_START
        self.X_END = X_END
        self.T_START = T_START
        self.T_END = T_END
        self.N = N
        self.M = self.N
        self.error = error
        self.i_max = 10000
        self.x = np.linspace(X_START, X_END, N)
        self.t = np.linspace(T_START, T_END, M)
        self.dx = self.x[1] - self.x[0]
        self.dt = self.t[1] - self.t[0]

    def initialize(self, c, c_integral, condition_initial, condition_border):
        self.c = c
        self.c_integral = c_integral
        self.condition_initial = condition_initial
        self.condition_border = condition_border
        self.u = np.empty((self.N, self.M))
        self.u[0] = condition_border(self.t)
        self.u.T[0] = condition_initial(self.x)
        self.__u11 = self.u[0][0]
        self.__u12 = self.u[0][1]
        self.__u21 = self.u[1][0]

    def calculate_u(self):
        print('Calculating...')
        for n in range(1, self.N):
          print(f'Progress: {round(n/(self.N-1)*100)}%')
          for m in range(1, self.M):
            self.__u11 = self.u[n-1][m-1]
            self.__u21 = self.u[n][m-1]
            self.__u12 = self.u[n-1][m]
            u_new = self.solve_newton(self.template, self.template_div, self.__u11, self.i_max)
            self.u[n][m] = u_new
        print('Done!')

    def template(self, u22):
        return 0.5*(self.__u12 - self.__u11 + u22 -self.__u21) / self.dt + 0.5*(self.c_integral(u22) - self.c_integral(self.__u12) + self.c_integral(self.__u21) - self.c_integral(self.__u11)) / self.dx

    def template_div(self, u22):
        return 0.5 / self.dt + 0.5*(self.c(u22)) / self.dx

    def solve_newton(self, f, f_div, x0, i_max):
        x = x0
        i = 0
        while abs(f(x)) > self.error and i < i_max:
            x += (-f(x) / f_div(x))
            i+=1
        return x


    def characteristics(self, x, x0=0, t0=0, first=True):
        if first:
            t0 = 0
            return 1/self.c(self.condition_initial(x0))*(x - x0)
        if not first:
            x0 = 0
            return 1/self.c(self.condition_border(t0))*x + t0

    def plot_characteristics(self, x_start, x_end, t_start, t_end, how_many):
        f, axs = plt.subplots(2, 2, figsize=(25, 10))
        f.suptitle('Characteristics plot', fontsize = 18)
        plt.subplot(1, 2, 1)
        plt.xlim(self.X_START, self.X_END)
        plt.ylim(t_start, t_end)
        plt.title('Характеристики при x0 = 0', fontsize = 18)
        plt.xlabel('x', fontsize = 18)
        plt.ylabel('t', fontsize = 18)
        x_for_char = np.linspace(x_start, x_end, how_many)
        t_for_char = np.linspace(t_start, t_end, how_many)
        for x0 in x_for_char:
            plt.plot(self.x, self.characteristics(self.x, x0 = x0))
        plt.subplot(1, 2, 2)
        plt.title('Характеристики при y0 = 0', fontsize = 18)
        plt.xlim(self.X_START, self.X_END)
        plt.ylim(t_start, t_end)
        plt.xlabel('x', fontsize = 18)
        plt.ylabel('t', fontsize = 18)
        for t0 in t_for_char:
            plt.plot(self.x, self.characteristics(self.x, t0 = t0, first = False))

    def plot_u(self, filename='OMM_Task_1', online = False):
        data = [go.Surface(x = self.x, y = self.t, z = self.u.T, colorscale = 'YlGnBu')]
        fig = go.Figure(data = data, layout = layout)

        if not online:
            return plotly.offline.iplot(fig, filename = filename)
        if online:
            return plotly.plotly.iplot(fig, filename = filename)

def c(u):
    return (2 + np.cos(u)) / (1 + (2*u + 1 + np.sin(u))**2)

def c_integral(u):
    return np.arctan(2*u + 1 + np.sin(u))

def condition_initial(x):
    return np.cos(np.pi*x/2)

def condition_border(t):
    return 1 + 0.5*np.arctan(t)

solver = TES(T_END = 20, N = 500, M = 500)
solver.initialize(c, c_integral, condition_initial, condition_border)
solver.plot_characteristics(0, 1, 0, 20, 20)
solver.calculate_u()
solver.plot_u(online = False)
plt.show()

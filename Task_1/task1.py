import numpy as np
import plotly
import plotly.graph_objs as go
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

layout = go.Layout(
    scene = dict(
    xaxis = dict(
        title='x',
        gridcolor="rgb(255, 255, 255)",
        zerolinecolor="rgb(255, 255, 255)",
        showbackground=True,
        backgroundcolor="rgb(200, 200, 230)"),

    yaxis = dict(
        title='y',
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
    #state_functions = False
    def __init__(self, X_START=0, X_END=1, T_START=0, T_END=1, N=100, M=100, error=0.0001):
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

    def initialize(self, c, c_integral, condition_initial, condition_border, heterogeneity, heterogeneity_div):
        self.c = c
        self.c_integral = c_integral
        self.condition_initial = condition_initial
        self.condition_border = condition_border
        self.heterogeneity = heterogeneity
        self.heterogeneity_div = heterogeneity_div
        self.u = np.empty((self.N, self.M))
        self.u[0] = condition_border(self.t)
        self.u.T[0] = condition_initial(self.x)
        self.__u11 = self.u[0][0]
        self.__u12 = self.u[0][1]
        self.__u21 = self.u[1][0]

    def calculate_u(self):
        print('Calculating...')
        for n in range(1, self.N):
            # clear_output(wait = True)
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
        f, axs = plt.subplots(2,2,figsize=(14,7))
        f.suptitle('Characteristics plot')
        plt.subplot(1, 2, 1)
        plt.xlim(self.X_START, self.X_END)
        plt.ylim(t_start, t_end)
        plt.title('First family')
        plt.xlabel('x')
        plt.ylabel('t')
        x_for_char = np.linspace(x_start, x_end, how_many)
        t_for_char = np.linspace(t_start, t_end, how_many)
        for x0 in x_for_char:
            plt.plot(self.x, self.characteristics(self.x, x0 = x0))
        plt.subplot(1, 2, 2)
        plt.title('Second family')
        plt.xlim(self.X_START, self.X_END)
        plt.ylim(t_start, t_end)
        plt.xlabel('x')
        plt.ylabel('t')
        for t0 in t_for_char:
            plt.plot(self.x, self.characteristics(self.x, t0 = t0, first = False))

    def plot_conditions(self):
        f, axs = plt.subplots(2,2,figsize=(14,7))
        f.suptitle('Conditions plot', )
        plt.subplot(1, 2, 1)
        plt.title('Initial Condition')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.plot(self.x, self.u.T[0][:])
        plt.subplot(1, 2, 2)
        plt.title('Border Condition')
        plt.xlabel('t')
        plt.ylabel('u')
        plt.plot(self.t, self.u[:][0])

    def visualise_evolution(self, filename = 'basic_animation'):
        fig = plt.figure(dpi = 200, figsize = (3, 2))
        ax = plt.axes(xlim=(self.X_START*0.9, self.X_END*1.1), ylim=(np.min(self.u)*0.9, np.max(self.u)*1.1))
        ax.set_title('Evolution of u(x, t)')
        ax.set_ylabel('u(x, t)', fontsize = 12)
        ax.set_xlabel('x', fontsize = 12)
        line, = ax.plot([], [], lw=2)
        textvar = fig.text(0.15, 0.15, '', fontsize = 20)

        def init():
            line.set_data([], [])
            return line,

        def animate(i):
            x = self.x
            y = self.u.T[i]

            textvar.set_text(f't = {round(self.t[i], 1)}')
            line.set_data(x, y)
            return line,

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=self.N, interval=5000/self.N, blit=True)
        plt.close(fig)
        anim.save(filename + '.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        return anim

    def plot_u(self, filename='OMM_Task_1', online = False):
        data = [go.Surface(x = self.x, y = self.t, z = self.u.T, colorscale = 'YlGnBu')]
        fig = go.Figure(data = data, layout = layout)

        if not online:
            return plotly.offline.iplot(fig, filename = filename)
        if online:
            return plotly.plotly.iplot(fig, filename = filename)

def c(u):
    return (2 + np.cos(u)) / (1 + 2*u + 1 + np.sin(u))**2

def c_integral(u):
    return np.arctan(2*u + np.sin(u) + 1)

def condition_initial(x):
    return np.cos(np.pi*x/2)

def condition_border(t):
    return 1 + 0.5*np.arctan(t)

def heterogeneity(u):
    return 0

def  heterogeneity_div(u):
    return 0

solver_small = TES(N = 100, M = 100)
solver_small.initialize(c, c_integral, condition_initial, condition_border, heterogeneity, heterogeneity_div)
solver_small.plot_characteristics(0, 1, 0, 20, 20)
solver = TES(T_END = 20, N = 500, M = 500)
solver.initialize(c, c_integral, condition_initial, condition_border, heterogeneity,  heterogeneity_div)
solver.plot_conditions()
plt.show()
solver.calculate_u()
solver.plot_u(online = False)

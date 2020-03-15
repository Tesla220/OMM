import time
import numpy as np
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot
from IPython.display import clear_output, HTML, display, Image

layout = go.Layout(
    scene = dict( aspectmode='cube', camera = dict(eye=dict(x=-2, y=1.5, z=1)),
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
        title='u(x, y, t)',
        gridcolor="rgb(255, 255, 255)",
        zerolinecolor="rgb(255, 255, 255)",
        showbackground=True,
        backgroundcolor="rgb(230, 230,200)",),),
    autosize=False,
    width=800, height=600,
    margin=dict(
        r=20, b=10,
        l=10, t=10),
    )
camera = dict(
    eye=dict(x=-2, y=2, z=1)
)

def TDMA(data, F):
    N = len(F)
    x = [0]*(N)
    A = []
    C = [data[0][0]]
    B = [data[0][1]]
    for i in range(1, N-1):
        A.append(data[i][i-1])
        C.append(data[i][i])
        B.append(data[i][i+1])
    A.append(data[-1][-2])
    C.append(data[-1][-1])

    alpha = [-B[0]/C[0]]*(N-1)
    beta = [F[0]/C[0]]*(N-1)

    for i in range(1, N-1):
        alpha[i] = -B[i]/(A[i-1]*alpha[i-1] + C[i])
        beta[i] = ((F[i] - A[i-1]*beta[i-1])/(A[i-1]*alpha[i-1] + C[i]))
    x[-1] = (F[-1] - A[-1]*beta[-1]) / (C[-1] + A[-1]*alpha[-1])

    for i in reversed(range(N-1)):
        x[i] = alpha[i]*x[i+1] + beta[i]

    return(np.array(x))

class HES():
    def __init__(self, X_START=0, X_END=2, Y_START=0, Y_END=1, T_START=0, T_END=20, N=5, M=5, J=5):
        self.X_START = X_START
        self.X_END = X_END
        self.Y_START = Y_START
        self.Y_END = Y_END
        self.T_START = T_START
        self.T_END = T_END
        self.N = N
        self.M = M
        self.J = J
        self.x = np.linspace(X_START, X_END, N)
        self.y = np.linspace(Y_START, Y_END, M)
        self.t = np.linspace(T_START, T_END, J)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dt = self.t[1] - self.t[0]

    def initialize(self, a=1, f=lambda x, y, t:0, fi=lambda x, y:0, alpha1x=0, alpha2x=0, beta2x=0, beta1x=0, mu1x=lambda y, t:0, mu2x=lambda y, t:0, alpha1y=0, alpha2y=0, beta2y=0, beta1y=0, mu1y=lambda x, t:0, mu2y=lambda x, t:0):
        self.a = a  # Коэффициент при операторе Лапласа
        self.f = f  # Функция "источника тепла" - неоднородность
        self.fi = fi  # Начальное условие
        self.alpha1x = alpha1x  # Коэффициент при левом условии Неймана для x
        self.alpha2x = alpha2x  # Коэффициент при правом условии Неймана для x
        self.beta1x = beta1x  # Коэффициент при левом условии Дирихле для x
        self.beta2x = beta2x  # Коэффициент при правом условии Дирихле для x
        self.mu1x = mu1x # Правая часть левого граничного условия для x
        self.mu2x = mu2x # Правая часть правого граничного условия для x
        self.alpha1y = alpha1y  # Коэффициент при левом условии Неймана для y
        self.alpha2y = alpha2y  # Коэффициент при правом условии Неймана для y
        self.beta1y = beta1y  # Коэффициент при правом условии Дирихле для y
        self.beta2y = beta2y  # Коэффициент при левом условии Дирихле для y
        self.mu1y = mu1y # Правая часть левого граничного условия для y
        self.mu2y = mu2y # Правая часть правого граничного условия для y
        self.u = np.zeros((self.J, self.N, self.M))  # Создание трехмерного массива значений u(x, y, t)
        self.u[0] = [[self.fi(x, y) for y in self.y] for x in self.x]  # Применение начального условия

    def calculate_layer(self, j):
        step_x = self.dx
        step_y = self.dt
        step_t = self.dt/2
        middle_layer = [np.array([0]*self.N)]*(self.M)
        for m in range(self.M):
            data = np.zeros((self.N, self.N))
            data[0][0] = self.beta1x - self.alpha1x/step_x
            data[0][1] = self.alpha1x/step_x
            data[-1][-2] = -self.alpha2x/step_x
            data[-1][-1] = self.alpha2x/step_x + self.beta2x

            for i in range(1, self.N - 1):
                data[i][i-1] = -self.a**2*step_t/step_x**2
                data[i][i] = (2*self.a**2*step_t/step_x**2 + 1)
                data[i][i+1] = -self.a**2*step_t/step_x**2

            F = [0]*self.N
            F[0], F[-1] = self.mu1x(self.y[m], self.t[j]+step_t), self.mu2x(self.y[m], self.t[j]+step_t)
            for k in range(1, self.N-1):
                F[k] = (self.f(self.x[k], self.y[m], self.t[j]+step_t)*step_t + self.u[j-1][k][m] + self.a**2*step_t/step_y**2 * (self.u[j-1][k-1][m] - 2*self.u[j-1][k][m] + self.u[j-1][k+1][m]))

            middle_layer[m] = TDMA(data, F)

        middle_layer = np.array(middle_layer).T
        new_layer = [np.array([0]*self.M)]*(self.N)

        for n in range(self.N):
            data = np.zeros((self.M, self.M))
            data[0][0] = self.beta1y - self.alpha1y/step_y
            data[0][1] = self.alpha1y/step_y
            data[-1][-2] = self.alpha2y/step_y
            data[-1][-1] = self.alpha2y/step_y + self.beta2y

            for i in range(1, self.M - 1):
                data[i][i-1] = -self.a**2*step_t/step_y**2
                data[i][i] = (2*self.a**2*step_t/step_y**2 + 1)
                data[i][i+1] = -self.a**2*step_t/step_y**2

            F = [0]*self.M
            F[0], F[-1] = self.mu1y(self.x[n], self.t[j]+2*step_t), self.mu2y(self.x[n], self.t[j]+2*step_t)
            for k in range(1, self.M-1):
                F[k] = (self.f(self.x[n], self.y[k], self.t[j]+2*step_t)*step_t + middle_layer[n][k] + self.a**2*step_t/step_x**2 * (middle_layer[n][k-1] - 2*middle_layer[n][k] + middle_layer[n][k+1]))

            new_layer[n] = TDMA(data, F)

        new_layer = np.array(new_layer)
        return new_layer

    def calculate_u(self):
        print('Calculating...')
        for j in range(1, self.J):
            #clear_output(wait = True)
            print(f'Progress: {round(j/(self.J-1)*100)}%')
            new_layer = self.calculate_layer(j)
            self.u[j] = new_layer
        print('Done!')

    def plot_state(self, n = 0, filename = '1', online = False):
        num = int(round(n/100*self.J, 0))
        if num>self.J-1:
            num = self.J-1

        data = [go.Surface(x = self.x, y = self.y, z = self.u[num].T, colorscale = 'YlGnBu')]
        fig = go.Figure(data = data, layout = layout)

        if online:
            display(plotly.plotly.iplot(fig, filename = filename))
        if not online:
            display(plotly.offline.iplot(fig, filename = filename))

    def plot_initial_state(self, filename = '2', online = True):
        self.plot_state(n = 0, filename = filename, online = online)

    def show_evolution(self):
        fig = go.Figure(layout = layout)
        surface = fig.add_surface(z=self.u[0].T, colorscale = 'YlGnBu')
        fig.layout.scene.zaxis.range = [np.min(self.u), np.max(self.u)]
        fig.layout.scene.yaxis.range = [self.Y_START, self.Y_END]
        fig.layout.scene.xaxis.range = [self.X_START, self.X_END]

        frames = []
        for j in range(self.J):
            frames.append(go.Frame(data=[{'type': 'surface','x': self.x, 'y': self.y, 'z': self.u[j].T}]))

        fig.frames = frames
        fig.layout.updatemenus = [{'buttons': [{'args': [None, {'frame': {'duration': 100, 'redraw': False}, 'fromcurrent': True, 'transition': {'duration': 100}}], 'label': 'Play', 'method': 'animate'}, {'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}], 'label': 'Pause', 'method': 'animate'}], 'direction': 'left', 'pad': {'r': 10, 't': 87}, 'showactive': False, 'type': 'buttons', 'x': 0.55, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'}]
        display(iplot(fig))

solver = HES(N = 50, M = 50, J = 50, T_END = 2)
a = 1
def f(x, y, t):
    return (y*t)**2
def fi(x, y):
    return np.cos(np.pi*x/4)*y*(1-y)
alpha1x = 1
alpha2x = 0
alpha1y = 0
alpha2y = 0
beta1x = 0
beta2x = 1
beta1y = 1
beta2y = 1
solver.initialize(a = a, f = f, fi = fi, alpha1x = alpha1x, alpha2x = alpha2x, alpha1y = alpha1y, alpha2y = alpha2y, beta1x = beta1x, beta2x = beta2x, beta1y = beta1y, beta2y = beta2y)

solver.plot_initial_state(online = False)
solver.calculate_u()
solver.plot_state(n = 50)
solver.plot_state(n = 100)
solver.show_evolution()

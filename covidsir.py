# -*- coding: latin-1 -*-
import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

# ******************************************************************************
# * Código para estimativa de propagação do Coronavírus
# * Autor: Sérgio Bastos    28/03/2020
# ******************************************************************************

# dados históricos
hist = pd.Series([5, 5, 5, 5, 7, 9, 14, 17, 27, 29, 31, 40, 46, 56, 66, 71])
inicio = '12/03/2020'

# População total N, gamma
N = 1000000
gamma = 0.5

popt_exponential, pcov_exponential = \
    scipy.optimize.curve_fit(lambda t, a, b: a * np.exp(b * t),
                             hist.index, hist, p0=(5, 0.3))

# Parâmetros para o modelo S-I-R
I0 = popt_exponential[0]
beta = gamma + popt_exponential[1]
R0 = beta / gamma
S0 = N - I0 - R0

t = np.linspace(0, 130, 130)


# Equações diferenciais do modelo S-I-R.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Vetor de condições iniciais
y0 = S0, I0, R0
# Integra as equações S-I-R ao longo do tempo, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plota os dados
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)


# As 3 curvas S(t), I(t) e R(t)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Suscetíveis')
ax.plot(t, I, 'm', alpha=0.5, lw=2, label='Infectados')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recuperados')

ax.set_xlabel('dias')
ax.set_ylabel('pessoas')
plt.title('Previsão do Coronavírus')
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
plt.ticklabel_format(axis="y", style="plain")
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()

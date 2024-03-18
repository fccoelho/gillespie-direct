"""
Milstein's method for solving multidimensional SDEs
"""
import copy
import numpy as np
import matplotlib.pyplot as plt


#
# np.random.seed(101)
# T=1; Delta=2**(-9); delta=Delta**2
# L=int(T/Delta); K=int(Delta/delta)
#
# X1 = np.zeros(L+1); X2 = np.zeros(L+1); X3 = np.zeros(L+1)
# Y2 = 0
#
# X1[0]=1; X2[0]=0.1; X3[0]=0.1
# for j in range(1,L+1):
#     Y1=0; Winc1=0; Winc2=0
#     for k in range(1,K+1):
#         dW1 = np.sqrt(delta)*np.random.randn(1)
#         dW2 = np.sqrt(delta)*np.random.randn(1)
#         Y1 += Y2*dW1
#         Y2 += dW2
#         Winc1 += dW1
#         Winc2 += dW2
#     X1[j] = X1[j-1] + X1[j-1]*X2[j-1]*Winc1 + \
#             X1[j-1]*(X2[j-1]**2)*0.5*(Winc1**2 - Delta) + \
#             0.3*X1[j-1]*X2[j-1]*Y1
#     X2[j] = X2[j-1] - (X2[j-1] - X3[j-1])*Delta + 0.3*X2[j-1]*Winc2 + \
#             0.9*X2[j-1]*0.5*(X2[j-1] - X3[j-1])*Delta
#     X3[j] = X3[j-1] + (X2[j-1] - X3[j-1])*Delta
#
# plt.plot(np.linspace(0,T,L+1), X1, 'r-')
# plt.plot(np.linspace(0,T,L+1), X2, 'b--')
# plt.plot(np.linspace(0,T,L+1), X3, 'k-.')
# plt.legend(("$X^1$", "$X^2$", "$X^3$"))
# plt.xlabel('t'); plt.ylabel('X',rotation=0)
# plt.show()

def milstein(tstart, ystart, tfinish, h, eqs, params):
    nsteps = int((tfinish - tstart) / h)
    sol = np.zeros((nsteps, len(eqs)))
    tvals = np.zeros(nsteps).reshape(-1, 1)
    sol[0] = ystart
    tvals[0] = tstart

    def deriv(f, x, params, h=0.0001):
        if x - h < 0:
            h = x
        return (f(x + h, *params) - f(x - h, *params)) / (2 * h)

    for step in range(nsteps - 1):
        R = np.random.normal(0, h)
        for i, (f1, f2) in enumerate(eqs):
            sol[step + 1, :] = sol[step, i] + h * f1(sol[step, i], *params) + f2(sol[step], *params) * R + 0.5 * f2(
                sol[step], *params) * deriv(f2, sol[step], params) * (R ** 2 - h)
            tvals[step + 1] = (tvals[step] + h)
    return np.hstack((tvals, sol))


determ = lambda x, b, d: (b - d) * x
stoc = lambda x, b, d: np.sqrt((b + d) * x)


def MilsteinExample(b=0.3,
                    d=0.27,
                    tfinish=50,
                    plots_at_a_time=10,
                    step=0.25,
                    clear_plot=True,
                    auto_update=True):
    print(r'<center>Solução pelo método de Milstein <br>da EDE:</center>')
    print(r'<center>$\mathrm{d}x(t) = (b - d)x \mathrm{d}t + \sqrt{((b+d)x)}\mathrm{d}W(t)$</center>')
    sol = milstein(0, 25, tfinish, step, [(determ, stoc)], (b, d))
    plt.plot(sol[:, 0], sol[:, 1:], color='cyan')
    # plt.title("Uma realização")
    # plt.show()
    exact = np.exp((b - d) * sol[:, 0]) * 25
    exact_plot = plt.plot(sol[:,0], exact, 'r--', label='Solução exata')
    sol_cp = copy.deepcopy(sol)
    for i in range(1, plots_at_a_time):
        sol2 = milstein(0, 25, tfinish, step, [(determ, stoc)], (b, d))
        # sol = [(v[0], sol[n][1] + v[1]) for n, v in enumerate(sol2)]
        sol_cp += sol2
        plt.plot(sol2[:,0], sol2[:,1:], alpha=0.3)

    # plt.plot(np.array([(i, v / plots_at_a_time) for i, v in sol_cp]), color='green', label="média")
    plt.plot(sol2[:, 0], (sol_cp[:, 1:]) / plots_at_a_time, color='green', label="média")
    # print(sol.shape, (sol_cp[:, 1:]) / plots_at_a_time)
    plt.legend()


if __name__ == "__main__":
    MilsteinExample()
    plt.show()

# -----------------------------------------------------------------------------
# Name:        gillespie.py
# Project:  Bayesian-Inference
# Purpose:     
#
# Author:      Flávio Codeço Coelho<fccoelho@gmail.com>
#
# Created:     2008-11-26
# Copyright:   (c) 2008 by the Author
# Licence:     GPL
# -----------------------------------------------------------------------------
__docformat__ = "restructuredtext en"

from numpy.random import uniform, multinomial, exponential, random
from numpy import arange, array, empty, zeros, log, isnan, nanmax, nan_to_num, ceil
import numpy as np
import time
import pdb
import copy
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu, default_ip=ti.i32, default_fp=ti.f32)


@ti.data_oriented
class Model:
    def __init__(self, vnames, rates, inits, tmat, propensity):
        '''
        Class representing a Stochastic Differential equation.
        
        :Parameters:
            - `vnames`: list of strings
            - `rates`: list of fixed rate parameters
            - `inits`: list of initial values of variables. Must be integers
            - `tmat`: Transition matrix; numpy array with shape=(len(inits),len(propensity))
            - `propensity`: list of lambda functions of the form: lambda r,ini: some function of rates ans inits.
        '''
        # check types
        for i in inits:
            if not isinstance(i, int):
                i = int(i)

        assert tmat.dtype == 'int64'
        self.vn = vnames
        self.nvars = len(self.vn)
        self.rates = ti.field(shape=(len(rates),), dtype=ti.f32)
        self.rates.from_numpy(array(rates))
        # self.inits = array(inits, dtype=np.int32)
        self.inits = ti.field(shape=(self.nvars,), dtype=ti.i32)
        self.inits.from_numpy(array(inits, dtype=np.int32))
        self.state = ti.field(shape=(self.nvars,), dtype=ti.i32)
        self.tm = ti.field(shape=(self.nvars, len(propensity)), dtype=ti.i32)
        self.tm.from_numpy(tmat)
        self.pv = propensity  # [compile(eq,'errmsg','eval') for eq in propensity]
        self.pvl = len(self.pv)  # length of propensity vector
        self.pv0 = ti.field(shape=(len(self.pv),), dtype=ti.f32)
        self.evseries = {}  # dictionary with complete time-series for each event type.
        self.time = None
        self.tmax = 0
        self.series = None
        self.steps = ti.field(dtype=ti.i32, shape=())
        self.viz = False  # if intermediate vizualization should be on

    def getStats(self):
        return self.time, self.series, self.steps, self.evseries

    # @ti.kernel
    def run(self, method='SSA', tmax=10, reps=1, viz=False, serial=False):
        '''
        Runs the model.
        
        :Parameters:
            - `method`: String specifying the solving algorithm. Currently only 'SSA'
            - `tmax`: duration of the simulation.
            - `reps`: Number of replicates.
            - `viz`: Boolean. Whether to show graph of each replicate during simulation
            - `serial`: Boolean. False to run replicate in parallel when more than one core is a vailable. True to run them serially (easier to debug).

        :Return:
            a numpy array of shape (reps,tmax,nvars)
        '''
        self.tmax = tmax
        self.res = ti.field(shape=(tmax, self.nvars, reps), dtype=ti.f32)
        # self.res = zeros((tmax, self.nvars, reps), dtype=np.float32)
        trng = arange(tmax, dtype=int)
        tvec = ti.ndarray(shape=(tmax,), dtype=ti.i32)
        tvec.from_numpy(trng)

        if method == 'SSA':
            for r in range(reps):
                res = self.GSSA(self.tmax, self.inits, self.rates, self.pv0)
                self.res[:, :, r] = res

                # if reps == 0:
                #     self.evseries = self.res[0][1]
                # else:
                #     self.evseries = [i[1] for i in self.res]

        elif method == 'SSAct':
            pass

        self.time = tvec
        self.series = self.res
        # self.steps=steps

    @ti.kernel
    def GSSA(self, tmax: ti.i32, ini: ti.template(),
             r: ti.template(), pv: ti.template()):
        '''
        Gillespie Direct algorithm
        '''

        tc: ti.f32 = 0  # current time
        tau: ti.f32 = 0
        last_tim = 0  # first time step of results
        evts = {i: [] for i in range(len(self.pv))}

        self.steps[None] = 0
        for i in range(self.res.shape[1]):
            self.res[0, i, 0] = ini[i]
            self.state[i] = ini[i]
        neg: ti.i32 = -1
        while tc <= tmax:
            i: ti.i32 = 0
            a0: ti.f32 = 0
            for j in ti.static(range(self.pvl)):
                pv[i] = self.pv[j](r, ini)
                a0 += pv[i]
                i += 1

            if np.sum(pv)[0] > 0:  # no change in state if pv is all zeros
                tau = self.calc_tau(a0)
                tc += tau
                tim = int(tm.ceil(tc))

                e = self.sample_event(pv, a0)  # event which will happen on this iteration


                ini += self.tm[:, e]
                if tc <= tmax:
                    evts[e].append(tc)
                # print tc, ini
                if tim <= tmax - 1:
                    self.steps += 1
                    #                if a0 == 0: break
                    if tim - last_tim > 1:
                        for j in range(last_tim, tim):
                            self.res[j, :] = self.res[last_tim, :]
                    self.res[tim, :] = ini
                else:
                    for j in range(last_tim, tmax):
                        self.res[j, :] = self.res[last_tim, :]
                    break
                last_tim = tim
            self.evseries = evts
            if a0 == 0:
                break  # breaks when no event has prob above 0

        return self.res
    @ti.func
    def calc_tau(self, a0):
        return (-1.0 / a0) * tm.log2(random())

    @ti.func
    def sample_event(self, pv, a0):
        probs = []
        for i in ti.static(range(self.pvl)):
            probs.append(ti.cast(pv[i]/a0, ti.f32))

        return multinomial(1, probs).nonzero()[0][0]

@ti.func
def p1(r, ini): return r[0] * ini[0] * ini[1]


@ti.func
def p2(r, ini): return r[1] * ini[1]


def main():
    vnames = ['S', 'I', 'R']
    ini = [500, 1, 0]
    rates = [.001, .1]
    tm = array([[-1, 0], [1, -1], [0, 1]])
    # prop=[lambda r, ini:r[0]*ini[0]*ini[1],lambda r,ini:r[0]*ini[1]]
    M = Model(vnames=vnames, rates=rates, inits=ini, tmat=tm, propensity=[p1, p2])
    t0 = time.time()
    M.run(tmax=80, reps=1000, viz=0, serial=1)
    print('total time: ', time.time() - t0)
    t, series, steps, evts = M.getStats()
    ser = series.mean(axis=0)
    # print evts, len(evts[0])
    from matplotlib.pyplot import plot, show, legend
    plot(t, ser, '-.')
    legend(M.vn, loc=0)
    show()


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()',sort=1,filename='gillespie.profile')
    main()

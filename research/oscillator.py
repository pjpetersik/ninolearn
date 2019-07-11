import numpy as np
import matplotlib.pyplot as plt

class recharge_oscillator(object):
    def __init__(self, r=0.25, alpha=0.125, b_0=2.5,
                 gamma=0.75, c=1., mu = 0.67):
        """
        Initialize the recharge oscillator in its non-dimensional form
        """
        self.r = r
        self.alpha = alpha
        self.b_0 = b_0
        self.mu = mu
        self.gamma = gamma
        self.c = c

        self.b = mu * b_0
        self.R = gamma * self.b - c


    def dhw_dt(self, hw, Te):
        return -self.r * hw - self.alpha * self.b * Te

    def dTe_dt(self, hw, Te):
        return self.R * Te + self.gamma * hw

    def integration_step(self, hw, Te, dt):
        dhw = self.dhw_dt(hw, Te) * dt
        hw_new = hw + dhw

        dTe = self.dTe_dt(hw, Te) * dt
        Te_new = Te + dTe

        return hw_new, Te_new











class limited_recharge_oscillator(object):
    def __init__(self, r=0.25, alpha=0.125, b_0=2.5,
                 gamma=0.75, c=1.,  mu = 0.6,
                 delta_s=0.3, d=1.):
        """
        Initialize the recharge oscillator in its non-dimensional form
        """
        self.r = r
        self.alpha = alpha
        self.b_0 = b_0
        self.mu = mu
        self.gamma = gamma
        self.c = c
        self.d = d
        self.delta_s = delta_s

        self.b = mu * b_0



    def h_e(self, hw, tauc, taue):
        return hw + (tauc + taue)/2

    def h_c(self, hw, tauc):
        return hw + tauc/2

    def tau_c(self,Te, Tc):
        return self.b * Tc #+ np.random.uniform(-0.1, 0.1)

    def tau_e(self,Te, Tc):
        return self.b * (Te - Tc)

    def dhw_dt(self, hw, Te, Tc):
        return -self.r * hw - self.alpha * (self.tau_c(Te, Tc) + self.tau_e(Te, Tc))/2

    def dTe_dt(self, hw, Te, Tc):
        tauc = self.tau_c(Te, Tc)
        taue = self.tau_e(Te, Tc)

        he = self.h_e(hw, tauc, taue)

        return  - self.c * Te + self.gamma * he #+ 0.1*tauc - 0.01* (hw + self.b * Te)**3

    def dTc_dt(self, hw, Te, Tc, t):
        tauc = self.tau_c(Te, Tc)
        taue = self.tau_e(Te, Tc)
        he = self.h_e(hw, tauc, taue)
        hc = self.h_c(hw, tauc)

        return  - self.d * Tc + self.delta_s * tauc + self.gamma * hc  #- 0.5*np.cos(np.pi*2 * t)#+ 5*np.random.uniform(-1, 1)#- 0.01*Tc**3



    def integration_step(self, hw, Te, Tc, t, dt):
        dhw = self.dhw_dt(hw, Te, Tc) * dt
        hw_new = hw + dhw

        dTe = self.dTe_dt(hw, Te, Tc) * dt
        Te_new = Te + dTe

        dTc = self.dTc_dt(hw, Te, Tc, t) * dt
        Tc_new = Tc + dTc

        # Diagonositcs
        tauc = self.tau_c(Te_new, Tc_new)
        taue = self.tau_e(Te_new, Tc_new)
        return hw_new, Te_new, Tc_new, tauc, taue

if __name__=="__main__":
    model = limited_recharge_oscillator()

    tmax = 120
    dt = 0.01
    t_arr =np.arange(0, tmax-dt, dt) / 6
    iters = int(tmax//dt)

    hw_arr = np.zeros(iters)
    Te_arr = np.zeros(iters)
    Tc_arr = np.zeros(iters)
    taue_arr = np.zeros(iters)
    tauc_arr = np.zeros(iters)

    hw_arr[0] = 0#np.random.uniform(-1.,1)
    Te_arr[0] = 0.0#np.random.uniform(-1.,1)
    Tc_arr[0] = 1#np.random.uniform(-1.,1)

    for i in range(1,iters):
        hw_arr[i], Te_arr[i], Tc_arr[i], tauc_arr[i], taue_arr[i] = model.integration_step(hw_arr[i-1], Te_arr[i-1], Tc_arr[i-1], t_arr[i-1], dt)

    plt.close("all")
    plt.plot(t_arr, hw_arr, 'k')
    plt.plot(t_arr, Te_arr, 'r')
    plt.plot(t_arr, taue_arr, 'orange')
    plt.plot(t_arr, Tc_arr, 'b')
    plt.plot(t_arr, tauc_arr, 'navy')
    plt.hlines(0,-10000,10000, linestyle='--')
    plt.xlim(0,max(t_arr))
   # plt.ylim(-3,3)
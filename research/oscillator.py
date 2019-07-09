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
                 gamma=0.75, c=1.0,  mu = 0.5,
                 delta_s=0.3, d=0.2):
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



    def h_e(self, hw, tau):
        return hw + tau

    def tau_c(self,Te, Tc):
        return self.b * (Te + Tc)/2 #+ Tc

    def tau_e(self,Te, Tc):
        return self.b * (Te - Tc)/2 -0.4*Tc


    def dhw_dt(self, hw, Te, Tc):
        tau = (self.tau_e(Te, Tc) + self.tau_c(Te, Tc))/2
        return -self.r * hw - self.alpha * tau

    def dTe_dt(self, hw, Te, Tc):
        tauc = self.tau_c(Te, Tc)
        taue = self.tau_e(Te, Tc)
        tau = taue + tauc
        he = self.h_e(hw, tau)

        return  - self.c * Te + self.gamma * he - 0.00*Tc - 0.01 * (hw + self.b * Te)**3 + self.delta_s * taue

    def dTc_dt(self, hw, Te, Tc, t):
        tauc = self.tau_c(Te, Tc)
        taue = self.tau_e(Te, Tc)
        hc = hw + tauc

        return  - self.d * Tc + self.delta_s * tauc# + 0.01*self.gamma * hc #+ np.sin(np.pi*t * 12) + np.random.uniform(-1, 1)# - 0.001*Tc**3 #



    def integration_step(self, hw, Te, Tc, t, dt):
        dhw = self.dhw_dt(hw, Te, Tc) * dt
        hw_new = hw + dhw

        dTe = self.dTe_dt(hw, Te, Tc) * dt
        Te_new = Te + dTe

        dTc = self.dTc_dt(hw, Te, Tc, t) * dt
        Tc_new = Tc + dTc

        return hw_new, Te_new, Tc_new

if __name__=="__main__":
    model = limited_recharge_oscillator()

    tmax = 120
    dt = 0.01
    t_arr =np.arange(0, tmax-dt, dt) / 6
    iters = int(tmax//dt)

    hw_arr = np.zeros(iters)
    Te_arr = np.zeros(iters)
    Tc_arr = np.zeros(iters)

    hw_arr[0] = np.random.uniform(-1.,1)
    Te_arr[0] = np.random.uniform(-1.,1)
    Tc_arr[0] = np.random.uniform(-1.,1)

    for i in range(1,iters):
        hw_arr[i], Te_arr[i], Tc_arr[i] = model.integration_step(hw_arr[i-1], Te_arr[i-1], Tc_arr[i-1], t_arr[i-1], dt)

    plt.close("all")
    plt.plot(t_arr, hw_arr, 'k')
    plt.plot(t_arr, Te_arr, 'r')
    plt.plot(t_arr, Tc_arr, 'b')
    plt.hlines(0,-10000,10000, linestyle='--')
    plt.xlim(0,max(t_arr))
   # plt.ylim(-3,3)
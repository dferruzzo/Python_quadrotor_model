# author: Diego Ferruzzo Correa
# date: 13/01/2024

from numpy import array, zeros, sin, cos, tan, matmul, tan, absolute, floor, arctan2, sqrt, vectorize
from numpy.linalg import inv
import matplotlib.pyplot as plt
from myfunctions import rk4
from simple_pid import PID

class Drone:
    def __init__(self):
        self.Ir = 1e-3       # motor's moment of inertia
        self.Ixx = 16.83e-3  # x-axis inertia
        self.Iyy = 16.38e-3  # y-axis inertia
        self.Izz = 28.34e-3  # z-axis inertia
        self.Omr = 1.0       # propeller's relative angular velocity
        self.l = 1.0         # distance from the rotor to the CG
        self.g = 9.8         # gravity constant
        self.m = 1.0         # vehicle total mass
        self.t0 = 0.0        # simulation initial time
        self.tf = 10.0       # simulation end time
        self.dt = 1e-3       # simulation step
        self.x0 = zeros(12)  # initial condition vector
        self.U = zeros(4)    # initial input Thrust and torques
        self.sol = Solution(self.t0, self.tf, self.dt, self.x0)
    
    def show_parameters(self):
        print('Ir  =', self.Ir)
        print('Ixx =', self.Ixx)
        print('Iyy =', self.Iyy)
        print('Izz =', self.Izz)
        print('Omr =', self.Omr)
        print('l   =', self.l)
        print('g   =', self.g)
        print('m   =', self.m)
        print('t0  =', self.t0)
        print('tf  =', self.tf)
        print('x0  =', self.x0)
        print('U  =', self.U)
        
    def translational_dynamics_rhs(self, euler, U1):
        # Return the right-hand of the translational dynamics
        phi = euler[0]
        theta = euler[1]
        psi = euler[2]
        ddxi = (cos(phi)*cos(psi)*sin(theta)+sin(phi)*sin(psi))*U1/self.m
        ddyi = (cos(phi)*sin(psi)*sin(theta)-cos(psi)*sin(phi))*U1/self.m
        ddzi = -self.g + cos(phi)*cos(theta)*U1/self.m
        return array([ddxi, ddyi, ddzi]).reshape(3, 1)

    def pqr_dynamics_rhs(self, omega, U2, U3, U4):
        # Return the right-hand side dynamics in pqr-coordinates
        p = omega[0]
        q = omega[1]
        r = omega[2]
        dp = ((self.Iyy-self.Izz)/self.Ixx)*q*r - \
            (self.Ir*self.Omr/self.Ixx)*q+U2/self.Ixx
        dq = ((self.Izz-self.Ixx)/self.Iyy)*p*r - \
            (self.Ir*self.Omr/self.Iyy)*p+U3/self.Iyy
        dr = ((self.Ixx-self.Iyy)/self.Izz)*p*q+U4/self.Izz
        return array([dp, dq, dr]).reshape(3, 1)

    def euler_dynamics_rhs(self, euler, omega):
        # Return the right-hand side dynamics for the Euler angles
        return self.W(euler).dot(omega)

    def W(self, euler):
        # returns W matrix
        # from numpy import array, tan, sin, cos
        phi = euler[0]
        theta = euler[1]
        psi = euler[2]
        return array([[1, (tan(theta)*sin(phi)).item(), (tan(theta)*cos(phi)).item()],
                      [0, cos(phi).item(), -sin(phi).item()],
                      [0, (sin(phi)/cos(theta)).item(), (cos(phi)/cos(theta)).item()]]).reshape(3, 3)

    def complete_dynamics_rhs(self, X, U):
        # output
        dX = zeros(12)
        # Inputs
        U1 = U[0]
        U2 = U[1]
        U3 = U[2]
        U4 = U[3]
        # states
        phi = X[9]
        theta = X[10]
        psi = X[11]
        euler = array([phi, theta, psi]).reshape(3, 1)
        #
        p = X[6]
        q = X[7]
        r = X[8]
        omega = array([p, q, r]).reshape(3, 1)
        #
        # derivative of pqr
        dpqr = self.pqr_dynamics_rhs(omega, U2, U3, U4)
        #
        # derivative of the translational velocity in the inertial frame
        dvi = self.translational_dynamics_rhs(euler, U1)
        #
        # derivative of euler angles
        deuler = self.euler_dynamics_rhs(euler, omega)
        #
        # forming the output right-hand side
        dX[0] = X[3]
        dX[1] = X[4]
        dX[2] = X[5]
        dX[3] = dvi[0]
        dX[4] = dvi[1]
        dX[5] = dvi[2]
        dX[6] = dpqr[0]
        dX[7] = dpqr[1]
        dX[8] = dpqr[2]
        dX[9] = deuler[0]
        dX[10] = deuler[1]
        dX[11] = deuler[2]
        return dX.reshape(X.shape)

    #def compute_trajectories(self, t0, tf, dt, x0, U):
    def compute_trajectories(self):
        #self.t0 = t0
        #self.tf = tf
        #self.dt = dt
        #self.x0 = x0
        #self.U = U
        t, x = rk4(lambda t, x: self.complete_dynamics_rhs(x, self.U),\
                   self.x0, self.t0, self.tf, self.dt)
        self.sol.t = t
        self.sol.x = x
        print('Trajectories computed!')

    def plot(self):
        t = self.sol.t
        xi = self.sol.x[:, 0]
        yi = self.sol.x[:, 1]
        zi = self.sol.x[:, 2]
        vxi = self.sol.x[:, 3]
        vyi = self.sol.x[:, 4]
        vzi = self.sol.x[:, 5]
        p = self.sol.x[:, 6]
        q = self.sol.x[:, 7]
        r = self.sol.x[:, 8]
        phi = self.sol.x[:, 9]
        theta = self.sol.x[:, 10]
        psi = self.sol.x[:, 11]
        # Figure 1 - Position
        fig1, (ax11, ax12, ax13) = plt.subplots(3, 1, sharex=True)
        ax11.plot(t, xi)
        ax11.set_title('Position')
        ax11.set_ylabel('$x_I(t)$')
        ax11.grid()
        ax12.plot(t, yi)
        ax12.set_ylabel('$y_I(t)$')
        ax12.grid()
        ax13.plot(t, zi)
        ax13.set_ylabel('$z_I(t)$')
        ax13.grid()
        ax13.set_xlabel('time')
        # Figure 2 - Velocity
        fig2, (ax21, ax22, ax23) = plt.subplots(3, 1, sharex=True)
        ax21.plot(t, vxi)
        ax21.set_title('Velocity')
        ax21.set_ylabel('$vx_I(t)$')
        ax21.grid()
        ax22.plot(t, vyi)
        ax22.set_ylabel('$vy_I(t)$')
        ax22.grid()
        ax23.plot(t, vzi)
        ax23.set_ylabel('$vz_I(t)$')
        ax23.grid()
        ax23.set_xlabel('time')
        # Figure 3 - pqr
        fig3, (ax31, ax32, ax33) = plt.subplots(3, 1, sharex=True)
        ax31.plot(t, p)
        ax31.set_title('pqr')
        ax31.set_ylabel('$p(t)$')
        ax31.grid()
        ax32.plot(t, q)
        ax32.set_ylabel('$q(t)$')
        ax32.grid()
        ax33.plot(t, r)
        ax33.set_ylabel('$r(t)$')
        ax33.grid()
        ax33.set_xlabel('time')
        # Figure 4 - Euler
        fig4, (ax41, ax42, ax43) = plt.subplots(3, 1, sharex=True)
        ax41.plot(t, phi)
        ax41.set_title('Euler angles')
        ax41.set_ylabel('$\phi(t)$')
        ax41.grid()
        ax42.plot(t, theta)
        ax42.set_ylabel('$\\theta(t)$')
        ax42.grid()
        ax43.plot(t, psi)
        ax43.set_ylabel('$\psi(t)$')
        ax43.grid()
        ax43.set_xlabel('time')
        #
        plt.show()


class Controlled_Drone():
    def __init__():
        drone = Drone()
    #

    def desired_euler_U1(self, vd, vx):
        vxer = vd[0] - vx[0]
        vyer = vd[1] - vx[1]
        vzer = vd[2] - vx[2]
        phi_d = arctan2(-vyer, sqrt(vxer**2+(self.g+vzer)**2))
        theta_d = arctan2(vxer, (self.g+vzer))
        psi_d = 0
        U1 = self.m*sqrt((vxer**2+vyer**2+(g+vzer)**2))
        return array([phi_d, theta_d, psi_d, U1])
    #

class Motor:
    # Brushless DC motor first order + delay model + propeller attached
    # # the delay is substituted by a second-order Padé approximation
    def __init__(self, Kp, tau_a):
        # local parameters
        self.Kp = Kp
        self.tau_a = tau_a
        self.t0 = 0.0  # simulation initial time
        self.tf = 10.0  # simulation end time
        self.dt = 1e-3  # simulation step
        self.x0 = zeros(1)    # initial condition vector
        self.U = zeros(1)  # zeros(0)   # initial input
        self.sol1 = Solution(self.t0, self.tf, self.dt, self.x0)
        self.sol2 = Solution(self.t0, self.tf, self.dt, self.x0)
    #

    def dynamics_rhs(self, omega_s, omega_e):
        # Motor brushless dynamics without delay
        # omega_s = angular velocity output
        # omega_e = angular velocity input
        return -(1/self.tau_a)*omega_s + (self.Kp/self.tau_a)*omega_e
    #

    def controlled_dynamics_rhs(self, omega_s, omega_r, Kp, Ki, Kd, T):
        # omega_s: angular velocity output
        # omega_r: angular velocity reference
        # it is used a close-loop PID
        pid = PID(Kp, Ki, Kd, T)
        pid_out = pid.output(omega_r-omega_s)
        return self.dynamics_rhs(omega_s, pid_out)
    #

    def compute_controlled_trajectory(self, t0, tf, dt, x0, U, Kp, Ki, Kd, T):
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.x0 = x0
        self.U = U
        t, x = rk4(lambda t, x: self.controlled_dynamics_rhs(
            x, U(t), Kp, Ki, Kd, T), self.x0, self.t0, self.tf, self.dt)
        self.sol2.t = t
        self.sol2.x = x
        print('Trajectories computed!')
    #

    def compute_trajectory(self, t0, tf, dt, x0, U):
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.x0 = x0
        self.U = U
        t, x = rk4(lambda t, x: self.dynamics_rhs(
            x, U(t)), self.x0, self.t0, self.tf, self.dt)
        self.sol1.t = t
        self.sol1.x = x
        print('Trajectories computed!')
    #

    def plot(self):
        plt.plot(self.sol1.t, self.sol1.x, label='open-loop trajectory')
        plt.plot(self.sol2.t, self.sol2.x, label='closed-loop PID trajectory')
        plt.xlabel('time')
        plt.grid()
        plt.legend()
        plt.show()
    #

    def output(self, v):
        # receive the input tension and outputs the angular velocity
        pass
    #

'''
class PID:
    # Discrete PID implementation
    def __init__(self, Kp, Kd, Ki, T):
        # PID parameters
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.T = T
        self.ck = zeros(2)
        self.ek = zeros(3)
        #

    def output(self, e):
        self.ek[0] = e
        self.ck[0] = self.ck[1] + self.Kd*self.ek[2] + (self.Ki*self.T**2-2*self.Kp*self.T-4*self.Kd)*self.ek[1]/(
            2*self.T) + (2*self.Kp*self.T+2*self.Kd+self.Ki*self.T**2)*self.ek[0]/(2*self.T)
        # updates
        self.ck[1] = self.ck[0]
        self.ek[2] = self.ek[1]
        self.ek[1] = self.ek[0]
        return self.ck[0]
'''

class Solution:
    def __init__(self, t0, tf, dt, x0):
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.x0 = x0
        self.N = absolute(floor((self.tf-self.t0)/self.dt)).astype(int)
        self.x = zeros((self.N+1, self.x0.size))
        self.t = zeros(self.N+1)

# -----------------------------------------------------------------------------------
# Main
#
"""
drone = Drone()

x0 = zeros(12)
x0[2] = 1  # initial z0
t0 = 0
tf = 10
dt = 1e-3
U1 = drone.m*drone.g+1e-5
U2 = 0
U3 = 0
U4 = 0
U = array([U1, U2, U3, U4])
drone.compute_trajectories(t0, tf, dt, x0, U)
drone.plot()

Kp = 0.9182
tau_a = 0.0569
motor = Motor(Kp, tau_a)

t0 = 0
tf = 1.0
dt = 1e-3
x0 = array([0])
def U(t):
    if t<=0.5:
        output = 0
    else:
        output=1    
    return output

#
K = 2.0
Ki = 1.5
Kd = 1.0
T = 1e-2

motor.compute_controlled_trajectory(t0, tf, dt, x0, U, K, Ki, Kd, T)
motor.compute_trajectory(t0, tf, dt, x0, U)
motor.plot()

t = motor.sol1.t
print(t)
print(vectorize(U)(t))
  

TODO
27/03/2023
[Done] Implement motor for u(t)
[Done] Implement Brushless DC motor model without delay
    [ ] Implement Brushless DC motor model with delay
[Done] Testing discrete PID with the DC motor model without delay
[Redo] I'm using a PID implementation https://pypi.org/project/simple-pid/ <<<<---- Need review the code (04/04/2023)
[ ] Plot in Motor class, include u(t) in plot
[ ] Add PID to the drone class
[ ] Implement closed loop control for the controlled-drone class
[ ] Add white noise to the pqr output in the drone class, it should have an ON/OFF flag
[ ] Implement attitude sensor for the drone class
[ ] Implement a Kalman Filter
[ ] Implement the linear model for the drone class
"""

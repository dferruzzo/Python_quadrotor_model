# author: Diego Ferruzzo Correa
# date: 13/01/2024

from numpy import array, zeros, sin, cos, tan, matmul, tan, absolute, floor, arctan2, sqrt, vectorize, concatenate
from numpy.linalg import inv
import matplotlib.pyplot as plt
from myfunctions import rk4
from simple_pid import PID

class Drone:
    def __init__(self) -> None:
        # parameters
        self.Ir = 1e-3          # motor's moment of inertia
        self.Ixx = 16.83e-3     # x-axis inertia
        self.Iyy = 16.38e-3     # y-axis inertia
        self.Izz = 28.34e-3     # z-axis inertia
        self.Omr = 1.0          # propeller's relative angular velocity
        self.l = 1.0            # distance from the rotor to the CG
        self.g = 9.8            # gravity constant
        self.m = 1.0            # vehicle total mass
        # simulation parameters
        self.t0 = 0.0           # simulation initial time
        self.tf = 10.0          # simulation end time
        self.dt = 1e-3          # simulation step
        # motors' parameters
        self.kf = 1e-3          # force constant
        self.km = 1e-3          # torque constant 
        # states
        self.x0 = zeros(12)     # initial condition vector      
        self.U0 = zeros(12)     # initial condition for motors' dynamcis
        self.U = array([0,0,0,0])       # initial input Thrust and torques
        # solution
        self.sol = Solution(self.t0, self.tf, self.dt, self.x0)

    def set_x0(self, x0):
        print('Initial condition changed from:')
        print('x0 =', self.x0)
        self.x0 = x0
        print('to:')
        print('x0 =', x0)

    def get_x0(self):
        return self.x0

    def set_U(self, U):
        print('Input changed from:')
        print('U =', self.U)
        print('to:')
        self.U = U
        print('U =', U)

    def get_U(self):
        print(self.U)
        return self.U
    
    def parameters(self):
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
        print('x = ', self.x)
        print('U  =', self.U)
            
    def dynamics(self,t,x,U):
        # complete dynamics
        vx = x[1]
        vy = x[3]
        vz = x[5]
        p = x[6]
        q = x[7]
        r = x[8]
        phi = x[9]
        theta = x[10]
        psi = x[11]
        U1 = U[0]
        U2 = U[1]
        U3 = U[2]
        U4 = U[3]
        dx = zeros(12)
        dx[0] = vx
        dx[1] = (cos(phi)*cos(psi)*sin(theta)+sin(phi)*sin(psi))*U1/self.m
        dx[2] = vy
        dx[3] = (cos(phi)*sin(psi)*sin(theta)-cos(psi)*sin(phi))*U1/self.m
        dx[4] = vz
        dx[5] = -self.g + cos(phi)*cos(theta)*U1/self.m
        dx[6] = ((self.Iyy-self.Izz)/self.Ixx)*q*r - (self.Ir*self.Omr/self.Ixx)*q+U2/self.Ixx
        dx[7] = ((self.Izz-self.Ixx)/self.Iyy)*p*r + (self.Ir*self.Omr/self.Iyy)*p+U3/self.Iyy
        dx[8] = ((self.Ixx-self.Iyy)/self.Izz)*p*q + U4/self.Izz
        dx[9] = p + q*tan(theta)*sin(phi) + r*tan(theta)*cos(phi)
        dx[10] = q*cos(phi) - r*sin(phi)
        dx[11] = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)
        return dx
        
    def compute_trajectories(self):
        # compute trajectories using Runge-Kutta 4 algorithm.
        t, x = rk4(lambda t, x: self.dynamics(t,x,self.U), self.x0, self.t0, self.tf, self.dt)
        self.sol.t = t
        self.sol.x = x
        print('Trajectories computed!')
        return self.sol

    def plots(self):
        t = self.sol.t
        xi = self.sol.x[:, 0]
        yi = self.sol.x[:, 2]
        zi = self.sol.x[:, 4]
        vxi = self.sol.x[:, 1]
        vyi = self.sol.x[:, 3]
        vzi = self.sol.x[:, 5]
        p = self.sol.x[:, 6]
        q = self.sol.x[:, 7]
        r = self.sol.x[:, 8]
        phi = self.sol.x[:, 9]
        theta = self.sol.x[:, 10]
        psi = self.sol.x[:, 11]
        fig1 = plt.figure()
        fig1.suptitle("Posição e Velocidade")
        plt.subplot(2,3,1)
        plt.subplots_adjust(wspace=0.5)
        plt.plot(t,xi)
        plt.grid()
        plt.legend(['$x(t)$'])
        plt.subplot(2,3,2)
        plt.plot(t,yi)
        plt.grid()
        plt.legend(['$y(t)$'])
        plt.subplot(2,3,3)
        plt.plot(t,zi)
        plt.grid()
        plt.legend(['z(t)'])
        plt.subplot(2,3,4)
        plt.plot(t,vxi)
        plt.grid()
        plt.legend(['$v_{x}$'])
        plt.subplot(2,3,5)
        plt.plot(t,vyi)
        plt.grid()
        plt.legend(['$v_{y}$'])
        plt.subplot(2,3,6)
        plt.plot(t,vzi)
        plt.grid()
        plt.legend(['$v_{z}$'])
        fig2 = plt.figure()
        fig2.suptitle("Velocidade angular e ângulos de Euler")
        plt.subplot(2,3,1)
        plt.subplots_adjust(wspace=0.5)
        plt.plot(t,p)
        plt.grid()
        plt.legend(['$p(t)$'])
        plt.subplot(2,3,2)
        plt.plot(t,q)
        plt.grid()
        plt.legend(['$q(t)$'])
        plt.subplot(2,3,3)
        plt.plot(t,r)
        plt.grid()
        plt.legend(['r(t)'])
        plt.subplot(2,3,4)
        plt.plot(t,phi)
        plt.grid()
        plt.legend(['$\phi(t)$'])
        plt.subplot(2,3,5)
        plt.plot(t,theta)
        plt.grid()
        plt.legend(['$\\theta(t)$'])
        plt.subplot(2,3,6)
        plt.plot(t,psi)
        plt.grid()
        plt.legend(['$\psi(t)$'])
        plt.show()

class Controlled_Drone(Drone):
    def __init__(self) -> None:
       pass

    def desired_euler_U1(self, vd, vx):
        vxer = vd[0] - vx[0]
        vyer = vd[1] - vx[1]
        vzer = vd[2] - vx[2]
        phi_d = arctan2(-vyer, sqrt(vxer**2+(self.g+vzer)**2))
        theta_d = arctan2(vxer, (self.g+vzer))
        psi_d = 0
        U1 = self.m*sqrt((vxer**2+vyer**2+(g+vzer)**2))
        return array([phi_d, theta_d, psi_d, U1])

class Motor():
    # Brushless DC motor first order + delay model + propeller attached
    # # the delay is substituted by a second-order Padé approximation
    def __init__(self) -> None:
        # local parameters
        self.kf = 1
        self.km = 1
        self.Kp = 1
        self.tau_a = 0.5
        self.t0 = 0.0  # simulation initial time
        self.tf = 10.0  # simulation end time
        self.dt = 1e-3  # simulation step
        self.x0 = zeros(1)    # initial condition vector
        self.U = zeros(1)  # zeros(0)   # initial input
        self.sol1 = Solution(self.t0, self.tf, self.dt, self.x0)
        self.sol2 = Solution(self.t0, self.tf, self.dt, self.x0)
    
    def dynamics(self, omega_s, omega_e):
        # Motor brushless dynamics without delay
        # omega_s = angular velocity output
        # omega_e = angular velocity input
        return -(1/self.tau_a)*omega_s + (self.Kp/self.tau_a)*omega_e
    
    def controlled_dynamics(self, omega_s, omega_r, Kp, Ki, Kd, T):
        # omega_s: angular velocity output
        # omega_r: angular velocity reference
        # it is used a close-loop PID
        pid = PID(Kp, Ki, Kd, T)
        pid_out = pid.output(omega_r-omega_s)
        return self.dynamics_rhs(omega_s, pid_out)
    
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
    
    def plot(self):
        plt.plot(self.sol1.t, self.sol1.x, label='open-loop trajectory')
        plt.plot(self.sol2.t, self.sol2.x, label='closed-loop PID trajectory')
        plt.xlabel('time')
        plt.grid()
        plt.legend()
        plt.show()
    
    def output(self, v):
        # receive the input tension and outputs the angular velocity
        pass
    
class Solution:
    def __init__(self, t0, tf, dt, x0):
        self.t0 = t0
        self.tf = tf
        self.dt = dt
        self.x0 = x0
        self.N = absolute(floor((self.tf-self.t0)/self.dt)).astype(int)
        self.x = zeros((self.N+1, self.x0.size))
        self.t = zeros(self.N+1)
        self.U = zeros((self.N+1, 4))

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
# Python quadrotor model

Modelo de drone quadrirrotor em python. O modelo implementado é:

$$
\begin{align}
    \ddot{x}_I &= (\cos\phi\cos\psi\sin\theta+\sin\phi\sin\psi)U_1/m\\
    \ddot{y}_I &= (\cos\phi\sin\psi\sin\theta-\cos\psi\sin\phi)U_1/m\\
    \ddot{z}_I &= -g + \cos\phi\cos\theta U_1/m\\
    \dot p &= ((I_{yy}-I_{zz})/I_{xx})qr - (I_r\Omega_r/I_{xx})q+U_2/I_{xx}\\
    \dot q &= ((I_{zz}-I_{xx})/I_{yy})pr + (I_r\Omega_r/I_{yy})p+U_3/I_{yy}\\
    \dot r &= ((I_{xx}-I_{yy})/I_{zz})pq+U_4/I_{zz}\\
    \dot{\eta}&=W\omega
\end{align}
$$

onde $\eta=(\phi,\theta,\psi)'$ representa os ângulos de Euler, $\omega=(p,q,r)'$ e

$$
W=\begin{pmatrix}
1 & \tan\theta\sin\phi & \tan\theta\cos\phi\\
0 & \cos\phi & -\sin\phi\\
0 & \sin\phi/\cos\theta & \cos\phi/\cos\theta
\end{pmatrix}.
$$

## Vetor de estado

Escolhemos como vetor de estado,

$$
\begin{align*}
x_1&=x\\
x_2&=v_{x}\\
x_3&=y\\
x_4&=v_y\\
x_5&=z\\
x_6&=v_z\\
x_7&=p\\
x_8&=q\\
x_9&=r\\
x_{10}&=\phi\\
x_{11}&=\theta\\
x_{12}&=\psi.
\end{align*}
$$

logo,

$$
\begin{align*}
\dot x_1&=x_2\\
\dot x_2&=(\cos x_{10}\cos x_{12}\sin x_{11}+\sin x_{10}\sin x_{12})U_1/m\\
\dot x_3&=x_4\\
\dot x_4&=(\cos x_{10}\sin x_{12}\sin x_{11}-\cos x_{12}\sin x_{10})U_1/m\\
\dot x_5&=x_6\\
\dot x_6&=-g + \cos x_{10}\cos x_{11} U_1/m\\
\dot x_7&=((I_{yy}-I_{zz})/I_{xx})x_8 x_9 - (I_r\Omega_r/I_{xx})x_8+U_2/I_{xx}\\
\dot x_8&=((I_{zz}-I_{xx})/I_{yy})x_7x_9 + (I_r\Omega_r/I_{yy})x_7+U_3/I_{yy}\\
\dot x_9&=((I_{xx}-I_{yy})/I_{zz})x_7x_8+U_4/I_{zz}\\
\dot x_{10}&=x_7 +  x_8\tan x_{11}\sin x_{10} + x_9\tan x_{11}\cos x_{10} \\
\dot x_{11}&=x_8\cos x_{10}   -x_9\sin x_{10} \\
\dot x_{12}&=x_8\sin x_{10}/\cos x_{11} + x_9 \cos x_{10}/\cos x_{11}.
\end{align*}
$$

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""
import scipy.integrate
import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy as sp
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train=True
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 50)
        self.output_layer = nn.Linear(50, 1)

    def forward(self, x):  # ,p,px):
        inputs = x
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        output = self.output_layer(layer1_out)
        return output

##
choice_load = input("Möchtest du ein State_Dict laden? y/n")
if choice_load == 'y':
    train=False
    filename = input("Welches State_Dict möchtest du laden?")
    net_phi = Net()
    net_v = Net()
    net_phi = net_phi.to(device)
    net_v = net_v.to(device)
    net_v.load_state_dict(torch.load('C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko_Kragarm_5.1\\saved_data\\'+filename+'_v'))
    net_phi.load_state_dict(torch.load(
        'C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko_Kragarm_5.1\\saved_data\\' + filename+'_phi'))
    net_v.eval()
    net_phi.eval()
##
# Definition der Parameter des statischen Ersatzsystems
Lb = float(input('Länge des Kragarms [m]: '))
E = 21  #float(input('E-Modul des Balkens [10^6 kNcm²]: '))
t = 10  #float(input('Querschnittshöhe des Balkens [cm]: '))
b = 10  #float(input('Querschnittsbreite des Balkens [cm]: '))
A = t*b
I = (b*t**3)/12
EI = E*I*10**-3
G = 80  #float(input('Schubmodul des Balkens [GPa]: '))
LFS = 1  #int(input('Anzahl Streckenlasten: '))
K = 5 / 6  # float(input(' Schubkoeffizient '))
Ln = np.zeros(LFS)
Lq = np.zeros(LFS)
s = [None] * LFS
normfactor = 10/(Lb**3/(K*A*G)+(11*Lb**5)/(120*EI))
x = np.linspace(0, Lb, 1000)
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=True).to(device), 1)

for i in range(LFS):
    # ODE als Loss-Funktion, Streckenlast
    Ln[i] = 0#float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
    Lq[i] = Lb#float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
    s[i] = str(normfactor)+"*x"#input(str(i + 1) + '. Streckenlast eingeben: ')

def h(x, j):
    return eval(s[j])

#Netzwerk System 1
def f(x, net_phi):
    u = net_phi(x)
    u_x = torch.autograd.grad(u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xx = torch.autograd.grad(u_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    u_xxx = torch.autograd.grad(u_xx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    ode = 0
    for i in range(LFS):
        ode += u_xxx + h(x - Ln[i], i) * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode

#Netzwerk für System 2
def g(x, net_v, net_phi):
    u = net_v(x)
    z = net_phi(x)
    u_x = torch.autograd.grad(u, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(u))[0]
    z_x = torch.autograd.grad(z, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(z))[0]
    z_xx = torch.autograd.grad(z_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(z))[0]
    ode = u_x - z/EI + z_xx/(K*A*G) * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode

qx = np.zeros(1000)
for i in range(LFS):
    qx = qx + (h(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1) - Ln[i], i).cpu().detach().numpy()).squeeze() * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])

Q0 = integrate.cumtrapz(qx, x, initial=0)
#Q0 = Q(0) = int(q(x)), über den ganzen Balken
qxx = qx * x
#M0 = M(0) = int(q(x)*x), über den ganzen Balken
M0 = integrate.cumtrapz(qxx, x, initial=0)
#Die nächsten Zeilen bis Iterationen geben nur die Biegelinie aus welche alle 10 Iterationen refreshed wird während des Lernens, man kann also den Lernprozess beobachten

f_anal = (-1 / 120 * normfactor * pt_x ** 5 + 1 / 6 * Q0[-1] * pt_x ** 3 - M0[-1] / 2 * pt_x ** 2) / EI + (
            1 / 6 * normfactor * (pt_x) ** 3 - Q0[-1] * pt_x) / (K * A * G)

##

if train:
    # Hyperparameter
    learning_rate = 0.01
    net_phi = Net()
    net_v = Net()
    net_phi = net_phi.to(device)
    net_v = net_v.to(device)
    mse_cost_function = torch.nn.MSELoss()  # Mean squared error
    optimizer = torch.optim.Adam([{'params': net_phi.parameters()}, {'params': net_v.parameters()}], lr=learning_rate)
    # Der Scheduler sorgt dafür, dass die Learning Rate auf einem Plateau mit dem factor multipliziert wird
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor=0.8)

    y1 = net_v(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device),
                               1))  # + net_phi(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1))
    fig = plt.figure()
    plt.grid()
    ax = fig.add_subplot()
    ax.set_xlim([0, Lb])
    ax.set_ylim([-11, 0])
    line1, = ax.plot(x, y1.cpu().detach().numpy())

    iterations = 1000000
    for epoch in range(iterations):
        optimizer.zero_grad()  # to make the gradients zero
        x_bc = np.linspace(0, Lb, 500)
        # linspace x Vektor zwischen 0 und 1, 500 Einträge gleichmäßiger Abstand
        # Zufällige Werte zwischen 0 und 1
        pt_x_bc = torch.unsqueeze(Variable(torch.from_numpy(x_bc).float(), requires_grad=True).to(device), 1)
        # unsqueeze wegen Kompatibilität
        pt_zero = Variable(torch.from_numpy(np.zeros(1)).float(), requires_grad=False).to(device)

        x_collocation = np.random.uniform(low=0.0, high=Lb, size=(1000 * int(Lb), 1))
        all_zeros = np.zeros((1000 * int(Lb), 1))

        pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
        pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
        ode_phi = f(pt_x_collocation, net_phi)
        ode_v = g(pt_x_collocation, net_v, net_phi)

        # Randbedingungen
        net_bc_out_phi = net_phi(pt_x_bc)
        net_bc_out_v = net_v(pt_x_bc)
        # ei --> Werte, die minimiert werden müssen
        u_x_phi = torch.autograd.grad(net_bc_out_phi, pt_x_bc, create_graph=True, retain_graph=True,
                                    grad_outputs=torch.ones_like(net_bc_out_phi))[0]
        u_xx_phi = torch.autograd.grad(u_x_phi, pt_x_bc, create_graph=True, retain_graph=True,
                                     grad_outputs=torch.ones_like(net_bc_out_phi))[0]
        u_xxx_phi = torch.autograd.grad(u_xx_phi, pt_x_bc, create_graph=True, retain_graph=True,
                                      grad_outputs=torch.ones_like(net_bc_out_phi))[0]

        #RB für Netzwerk 1
        e1_phi = net_bc_out_phi[0]
        e2_phi = u_x_phi[0] + M0[-1]
        e3_phi = u_x_phi[-1]
        e4_phi = u_xx_phi[0] - Q0[-1]
        e5_phi = u_xx_phi[-1]


        #RB für Netzwerk 2
        e1_v = net_bc_out_v[0]

        #Alle e's werden gegen 0-Vektor (pt_zero) optimiert.

        mse_bc_phi = mse_cost_function(e1_phi, pt_zero) + 1/normfactor * mse_cost_function(e2_phi, pt_zero) + mse_cost_function(e3_phi, pt_zero) + 1/normfactor * mse_cost_function(e4_phi, pt_zero) + mse_cost_function(e5_phi, pt_zero)
        mse_ode_phi = mse_cost_function(ode_phi, pt_all_zeros)
        mse_bc_v = mse_cost_function(e1_v, pt_zero)
        mse_ode_v = mse_cost_function(ode_v, pt_all_zeros)

        loss = mse_ode_v + mse_ode_phi + mse_bc_v + mse_bc_phi

        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        with torch.autograd.no_grad():
            if epoch % 10 == 9:
                print(epoch, "Traning Loss:", loss.data)
                plt.grid()
                net_out = net_v(pt_x)
                net_out_v = net_out
                net_out_v_cpu = net_out_v.cpu().detach().numpy()
                err = torch.norm(net_out_v - f_anal, 2)
                print(f'Error = {err}')
                if err < 0.1 * Lb:
                    print(f"Die L^2 Norm des Fehlers ist {err}.\nStoppe Lernprozess")
                    break
                line1.set_ydata(net_out_v_cpu)
                fig.canvas.draw()
                fig.canvas.flush_events()

##
if choice_load == 'n':
    choice_save = input("Möchtest du die Netzwerkparameter abspeichern? y/n")
    if choice_save == 'y':
        filename = input("Wie soll das State_Dict heißen?")
        torch.save(net_v.state_dict(),'C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko_Kragarm_5.1\\saved_data\\'+filename+'_v')
        torch.save(net_phi.state_dict(),
                   'C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko_Kragarm_5.1\\saved_data\\' + filename+'_phi')
##
x = np.linspace(0, Lb, 1000)
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=True).to(device), 1)

v_out = net_v(pt_x)
phi_out = net_phi(pt_x)
v_out_x = torch.autograd.grad(v_out, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_out))[0]
v_out_xx = torch.autograd.grad(v_out_x, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_out))[0]
v_out_xxx = torch.autograd.grad(v_out_xx, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_out))[0]


v_out_x = v_out_x.cpu().detach().numpy()
v_out_xx = v_out_xx.cpu().detach().numpy()
v_out_xxx = v_out_xxx.cpu().detach().numpy()


#Netzwerk 1 Kompatibilität Numpy Array
phi_out_cpu = phi_out.cpu()
phi_out = phi_out_cpu.detach()
phi_out = phi_out.numpy()

#Netzwerk 2 Kompatibilität Numpy Array
v_out_cpu = v_out.cpu()
v_out = v_out_cpu.detach()
v_out = v_out.numpy()

fig = plt.figure()


plt.subplot(2, 2, 1)
plt.title('$v_{ges}$ Auslenkung')
plt.xlabel('')
plt.ylabel('[cm]')
plt.plot(x, v_out)
plt.plot(x, ((-1/120 * normfactor * x**5 + Q0[-1]/6 * x**3 - M0[-1]/2 * x**2)/EI + (1/6 * normfactor* x**3 - Q0[-1] * x)/(K*A*G)))
#plt.plot(x, (-1/24 *x**4-np.sin(x)+(Q0[-1]-1)/6 *x**3 - M0[-1]/2 * x**2 +x)/EI - (0.5*x**2 - np.sin(x) - (Q0[-1]-1)*x)/(K*A*G))
plt.grid()

plt.subplot(2, 2, 2)
plt.title('$\phi$ Neigung')
plt.xlabel('')
plt.ylabel('$10^{-2}$')
plt.plot(x, phi_out/EI)
plt.plot(x, (-1/24 * normfactor * x**4 + 0.5 * Q0[-1] * x**2 - M0[-1] * x)/EI)
#plt.plot(x, (-1/6 *x**3-np.cos(x)+(Q0[-1]-1)/2 * x**2 - M0[-1]*x+1)/EI)
plt.grid()


plt.subplot(2, 2, 3)
plt.title('Schubwinkel $\gamma$')
plt.xlabel('')
plt.ylabel('$(10^{-2})$')
plt.plot(x, (v_out_x - phi_out)/(K*A*G))
plt.plot(x, (normfactor * 0.5 * x**2 - Q0[-1])/(K*A*G))
#plt.plot(x, (x-np.cos(x)-(Q0[-1]-1))/(K*A*G))
plt.grid()




plt.show()
##
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 5)
        self.hidden_layer2 = nn.Linear(5, 15)
        self.hidden_layer3 = nn.Linear(15, 50)
        self.hidden_layer4 = nn.Linear(50, 50)
        self.hidden_layer5 = nn.Linear(50, 50)
        self.hidden_layer6 = nn.Linear(50, 25)
        self.hidden_layer7 = nn.Linear(25, 15)
        self.output_layer = nn.Linear(15, 1)

    def forward(self, x):  # ,p,px):
        inputs = x
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        layer6_out = torch.tanh(self.hidden_layer6(layer5_out))
        layer7_out = torch.tanh(self.hidden_layer7(layer6_out))
        output = self.output_layer(layer7_out)
        return output

##
choice_load = input("Möchtest du ein State_Dict laden? y/n")
if choice_load == 'y':
    train=False
    filename = input("Welches State_Dict möchtest du laden?")
    net = Net()
    net = net.to(device)
    net.load_state_dict(torch.load('C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko NN Kragarm 5.3\\saved_data\\'+filename))
    net.eval()
##
# Hyperparameter
learning_rate = 0.01
net_phi = Net()
net_v = Net()
net_phi = net_phi.to(device)
net_v = net_v.to(device)
mse_cost_function = torch.nn.MSELoss()  # Mean squared error
optimizer = torch.optim.Adam([{'params': net_phi.parameters()}, {'params': net_v.parameters()}], lr=learning_rate)
#Der Scheduler sorgt dafür, dass die Learning Rate auf einem Plateau mit dem factor multipliziert wird
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor= 0.8)

# Definition der Parameter des statischen Ersatzsystems
Lb = float(input('Länge des Kragarms [m]: '))
E = 21  #float(input('E-Modul des Balkens [10^6 kNcm²]: '))
h = 10  #float(input('Querschnittshöhe des Balkens [cm]: '))
b = 10  #float(input('Querschnittsbreite des Balkens [cm]: '))
A = h*b
I = (b*h**3)/12
EI = E*I*10**-3
G = 80  #float(input('Schubmodul des Balkens [GPa]: '))
LFS = 1  #int(input('Anzahl Streckenlasten: '))
K = 5 / 6  # float(input(' Schubkoeffizient '))
Ln = np.zeros(LFS)
Lq = np.zeros(LFS)
s = [None] * LFS
normfactor = 10/(Lb**3/(K*A*G)+(11*Lb**5)/(120*EI))

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


x = np.linspace(0, Lb, 1000)
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=True).to(device), 1)
qx = np.zeros(1000)
for i in range(LFS):
    qx = qx + (h(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1) - Ln[i], i).cpu().detach().numpy()).squeeze() * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])

Q0 = integrate.cumtrapz(qx, x, initial=0)
#Q0 = Q(0) = int(q(x)), über den ganzen Balken
qxx = qx * x
#M0 = M(0) = int(q(x)*x), über den ganzen Balken
M0 = integrate.cumtrapz(qxx, x, initial=0)
#Die nächsten Zeilen bis Iterationen geben nur die Biegelinie aus welche alle 10 Iterationen refreshed wird während des Lernens, man kann also den Lernprozess beobachten
y1 = net_v(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1)) #+ net_phi(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1))
fig = plt.figure()
plt.grid()
ax = fig.add_subplot()
ax.set_xlim([0, Lb])
ax.set_ylim([-30, 0])
line1, = ax.plot(x, y1.cpu().detach().numpy())
f_anal = (-1 / 120 * normfactor * pt_x ** 5 + 1 / 6 * Q0[-1] * pt_x ** 3 - M0[-1] / 2 * pt_x ** 2) / EI + (
            1 / 6 * normfactor * (pt_x) ** 3 - Q0[-1] * pt_x) / (K * A * G)
##
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
    net_bc_out_S = net_v(pt_x_bc)
    # ei --> Werte, die minimiert werden müssen
    u_x_phi = torch.autograd.grad(net_bc_out_phi, pt_x_bc, create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(net_bc_out_phi))[0]
    u_xx_phi = torch.autograd.grad(u_x_phi, pt_x_bc, create_graph=True, retain_graph=True,
                                 grad_outputs=torch.ones_like(net_bc_out_phi))[0]
    u_xxx_phi = torch.autograd.grad(u_xx_phi, pt_x_bc, create_graph=True, retain_graph=True,
                                  grad_outputs=torch.ones_like(net_bc_out_phi))[0]
    u_x_v = torch.autograd.grad(net_bc_out_v, pt_x_bc, create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(net_bc_out_v))[0]

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

    loss = 3*mse_ode_v + 3*mse_ode_phi + mse_bc_v + mse_bc_phi

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
        torch.save(net.state_dict(),'C:\\Users\\Administrator\\Desktop\\Uni\\Master\\Masterarbeit\\Timoshenko NN Kragarm 5.3\\saved_data\\'+filename)
##
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=True).to(device), 1)

pt_u_out_v = net_v(pt_x)
pt_u_out = net_phi(pt_x)
w_x = torch.autograd.grad(pt_u_out, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(pt_u_out))[0]
w_xx = torch.autograd.grad(w_x, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(pt_u_out))[0]
w_xxx = torch.autograd.grad(w_xx, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(pt_u_out))[0]

ws_x = torch.autograd.grad(pt_u_out_s, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(pt_u_out))[0]

w_x = w_x.cpu().detach().numpy()
w_xx = w_xx.cpu().detach().numpy()
w_xxx = w_xxx.cpu().detach().numpy()

ws_x = ws_x.cpu().detach().numpy()

#Netzwerk 1 Kompatibilität Numpy Array
u_out_cpu = pt_u_out.cpu()
u_out = u_out_cpu.detach()
u_out = u_out.numpy()

#Netzwerk 2 Kompatibilität Numpy Array
s_out_cpu = pt_u_out_s.cpu()
s_out = s_out_cpu.detach()
s_out = s_out.numpy()

fig = plt.figure()


plt.subplot(3, 2, 1)
plt.title('$v_{ges}$ Auslenkung')
plt.xlabel('')
plt.ylabel('[cm]')
plt.plot(x, s_out)
plt.plot(x, (-1/120 *x**5 + 25/12 * x**3 - 41.67/2 * x**2)/EI + (1/6 * x**3 - 12.5*x)/(K*A*G))
#plt.plot(x, (-1/24 *x**4-np.sin(x)+(Q0[-1]-1)/6 *x**3 - M0[-1]/2 * x**2 +x)/EI - (0.5*x**2 - np.sin(x) - (Q0[-1]-1)*x)/(K*A*G))
plt.grid()

plt.subplot(3, 2, 3)
plt.title('$\phi$ Neigung')
plt.xlabel('')
plt.ylabel('$10^{-2}$')
plt.plot(x, u_out/EI)
plt.plot(x, (-1/24 * x**4 +25/4 *x**2 - 41.67*x)/EI)
#plt.plot(x, (-1/6 *x**3-np.cos(x)+(Q0[-1]-1)/2 * x**2 - M0[-1]*x+1)/EI)
plt.grid()

plt.subplot(3, 2, 5)
plt.title('$\kappa$ Krümmung')
plt.xlabel('Meter')
plt.ylabel('$(10^{-4})$[1/cm]')
plt.plot(x, w_x/EI)
plt.plot(x, (-1/6 * x**3 + 12.5 * x - 41.67)/EI)
#plt.plot(x, (-0.5*x**2+np.sin(x)+(Q0[-1]-1)*x-M0[-1])/EI)
plt.grid()

plt.subplot(3, 2, 2)
plt.title('Schubwinkel $\gamma$')
plt.xlabel('')
plt.ylabel('$(10^{-2})$')
plt.plot(x, (ws_x - (u_out)/EI))
plt.plot(x, (0.5*x**2 - 12.5)/(K*A*G))
#plt.plot(x, (x-np.cos(x)-(Q0[-1]-1))/(K*A*G))
plt.grid()


plt.subplot(3, 2, 4)
plt.title('q(x) Test')
plt.xlabel('')
plt.ylabel('$kN$')
plt.plot(x, (-w_xxx))
plt.plot(x, x)
#plt.plot(x, 1+np.sin(x))
plt.grid()



plt.show()
##
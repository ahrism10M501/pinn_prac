# Burger's Equation
# ∂u/∂t = v*((∂^2)u/∂(x^2)) - U*(∂u / ∂x)
# x ∈ [-1, 1], t ∈ [0, 1]
# u(0, x) = -sin(pi*x)
# u(t, -1) = u(t, 1) = 0

# u(t, x): 특정 시간 t와 위치 x에서 유체의 속도
# v : 점성 게수, 유체의 끈적이는 정도

import torch
import torch.nn as nn
import numpy as np

from pinn.model import pinn
import pinn.utils as utils
import matplotlib.pyplot as plt

from tqdm import tqdm
import os

os.makedirs('./params', exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"current device: {device}")

def collocation_points(*args,sampling=None):

    grids = torch.meshgrid(*args, indexing='ij')
    coordinates = torch.stack(grids, dim=len(args)).reshape(-1, len(args))
    
    if sampling:
        coordinates_t0 = coordinates[coordinates[:, 0] == coordinates[0,0]]
        #coordinates_te = coordinates[coordinates[:, 0] == coordinates[-1,0]]

        coordinates_sub = coordinates[((coordinates[:, 0] != coordinates[0,0]) & (coordinates[:, 0] != coordinates[-1,0]))]
        
        coordinates = coordinates_t0 #torch.vstack([coordinates_t0,coordinates_te])

        ninstances = int(len(coordinates_sub)/sampling[1])
        #print(len(coordinates_te),len(coordinates_t0),ninstances*sampling[1])
        for i in range(sampling[1]):
            idx = np.random.randint(low=0, high=ninstances, size=int(sampling[0]*ninstances)) 
            coordinates = torch.vstack([coordinates,coordinates_sub[i*ninstances:(i+1)*ninstances][idx]])
  
    return coordinates

# 구하고자 하는 식 정의 (PINN의 핵심)
def fwd_gradients(obj, x):
    # 신경망 출력(obj)을 입력(x)로 미분
    dummy = torch.ones_like(obj)
    derivative = torch.autograd.grad(obj, x, dummy, create_graph= True)[0]
    return derivative

def burgers_equation(u, tx):
    u_tx = fwd_gradients(u, tx) # tx는 [t, x]
    u_t = u_tx[:, 0:1] # 첫번째 열, 시간에 대한 편미분
    u_x = u_tx[:, 1:2] # 두번째 열, 공간에 대한 편미분
    u_xx = fwd_gradients(u_x, tx)[:, 1:2] # 2차 미분 후 공간에 대한 값만 가져오기

    # ∂u/∂t = v*((∂^2)u/∂(x^2)) - U*(∂u / ∂x)
    # ∂u/∂t + U*(∂u / ∂x) - v*((∂^2)u/∂(x^2)) = 0
    # 점성이 0.01/pi 인 버거스 방정식 모델 성능 실험
    e = u_t + u*u_x - (0.01/torch.pi)*u_xx
    return e

def relative_l2_error(u,w):
    zl2 = np.sqrt(np.sum((u-w)**2, axis=1))
    ul2 = np.sqrt(np.sum(u**2, axis=1))
    return zl2/ul2

nt = 100
nx = 64
t = torch.linspace(0, 1, nt)
x = torch.linspace(-1, 1, nx)

# 일반조건에 대한 데이터 생성
cdata = collocation_points(t, x).to(device)

# 경계조건에 대한 데이터 생성
icdata = collocation_points(t[0], x).to(device)
bcdata = collocation_points(t[[50, 90]], x[[0, -1]]).to(device)

# 실제 초기조건에 대한 제약 조건 만들어주기
u_init = -torch.sin(torch.pi*x).reshape(-1, 1).to(device)
plt.plot(x, u_init.cpu().numpy().reshape(-1))
plt.title(r'$u(0,x)=-sin(\pi x)$')
plt.show()

# Neural Network로 구해보기
hidden = 4
nodes = 32

layers = [2] + hidden * [nodes] + [1]
model = pinn(layers).to(device)

# hyper params
lr = 1e-3
epochs = 100000
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
ls = 10

cdata_req = cdata.clone()
cdata_req.requires_grad = True

pbar = tqdm(range(epochs))

for epoch in pbar:
    optimizer.zero_grad()

    u0_pred = model(icdata) # t=0 일 때 예측값
    ubc_pred = model(bcdata)
    u_pred = model(cdata_req)

    loss_col = torch.mean(burgers_equation(u_pred, cdata_req)**2)
    loss_init = torch.mean((u0_pred- u_init)**2)
    loss_bc = torch.mean(ubc_pred**2)

    loss = loss_col + loss_init + loss_bc

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        pbar.set_postfix({'loss': f'{loss.item():.6f}', 'best_ls': f'{ls:.6f}'})

        if loss.item() < ls:
            ls = loss.item()
            torch.save(model.state_dict(), './params/bg.pt')

print(f"mse: {ls}")

# 학습 후 검증
nt = 101
nx = 256
t = torch.linspace(0, 1, nt)
x = torch.linspace(-1, 1, nx)
cdata = collocation_points(t, x).to(device)

model.load_state_dict(torch.load('./params/bg.pt', map_location=device))
model.eval()

with torch.no_grad():
    u_pred = model(cdata).cpu().numpy()

# reference : 수치해석적으로 만들어진 계산값
u_sol = np.load('./sols/bgsol.npy')

rel_err = relative_l2_error(u_sol, u_pred.reshape(nt, nx))

plt.figure(figsize=(8,4))
plt.plot(t, 100 * rel_err, "--", color="black")
plt.title(f"Relative Error")
plt.xlabel("Time (t)")
plt.ylabel("%")
plt.grid(True, alpha=0.3)
plt.show()

fig = plt.figure(figsize=(8, 5))
gs = fig.add_gridspec(2, 3)

ax1 = fig.add_subplot(gs[0, :])
im1 = ax1.imshow(u_pred.reshape(nt, nx).T, interpolation='nearest', cmap='bwr', extent=[t.min(), t.max(), x.min(), x.max()], origin='lower', aspect='auto')

cbar = plt.colorbar(im1)
cbar.set_ticks([-1, 0, 1])

ax1.set_title('u(t, x)')
ax1.set_xlabel('t')
ax1.set_ylabel('x')

x_np = x.numpy()
t_np = t.numpy()
u_pred_2d = u_pred.reshape(nt, nx)

n1, n2, n3 = 25, 50, 90
for i, n in enumerate([n1, n2, n3]):
    ax = fig.add_subplot(gs[1, i])
    ax.plot(x_np, u_sol[n], color='blue')
    ax.plot(x_np, u_pred_2d[n], '--')
    ax.set_title(rf'$t={t_np[n]:.1f}$')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('x')
    ax.legend(labels=[r'$u_{FEM}$', r'$u_{PINN}$'])

plt.tight_layout()
plt.show()
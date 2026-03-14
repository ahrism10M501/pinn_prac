# m*(d^2y(t)/dt^2)+u*(dy(t)/dt)+ky(t) = 0
# m: mass, k: spring constant, u: friction coefficient
# m=1, k=101, u=2
# y(0) = 1

# 스프링을 1만큼 당겼다가 놓았을 때, 마찰 때문에 진동이 줄어들며 멈추는 과정을 신경망이 학습한다.

import torch
import torch.nn as nn

from pinn.model import pinn
import pinn.utils as utils
import matplotlib.pyplot as plt

from tqdm import tqdm
import os
os.makedirs('./params', exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"current device: {device}")

# 숫식 파라미터 정의
params = {
    "m":1,
    "mu":2,
    "k":101
}

# 구하고자 하는 식 정의 (PINN의 핵심)
def fwd_gradients(obj, x):
    # 신경망 출력(obj)을 입력(x)로 미분
    dummy = torch.ones_like(obj)
    derivative = torch.autograd.grad(obj, x, dummy, create_graph= True)[0]
    return derivative

def harmonic_oscillator(u, t, p):
    u_t = fwd_gradients(u, t) # 1차 미분
    u_tt = fwd_gradients(u_t, t) # 2차 미분
    e = p["m"]* u_tt + p["mu"]*u_t + p["k"]*u 
    return e

# 수식으로 실제 해를 직접 구해보기
d = params["mu"]/(2*params["m"])
w = ((params["k"]/params["m"])-d**2)**0.5
t = torch.linspace(0, 1, 50).to(device).reshape(-1, 1)

y_sol = torch.exp(-t)*torch.cos(w*t)
y_init = y_sol[0].item()
print(f"w : {w}, y0 : {y_init}")

# Neural Network로 구해보기
hidden = 4
nodes = 32

layers = [1] + hidden * [nodes] + [1]
model = pinn(layers).to(device)

# hyper params
lr = 2e-3
epochs = 10000
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# loss 가중치
beta = 10

ls = 10

t_req = t.clone()
t_req.requires_grad = True
t_zero = torch.zeros(1, 1).to(device).requires_grad_(True)

pbar = tqdm(range(epochs))

for epoch in pbar:
    optimizer.zero_grad()

    y0_pred = model(t_zero) # t=0 일 때 예측값
    v0_pred = fwd_gradients(y0_pred, t_zero) # 초기 속도 예측값

    y_pred = model(t_req) # 모든 시간일 때 예측값
    
    # MSE Loss
    # 물리 법칙 위반 정도
    loss_col = torch.mean(harmonic_oscillator(y_pred, t_req, params)**2)

    # 초기 위치값 오류
    loss_init_pos = torch.mean((y0_pred - y_init)**2)

    # 3. 초기 속도 Loss (y'(0) = -1) - 실제 해와 일치시키기 위함
    # 실제 해 y = e^-t * cos(10t)를 미분해서 t=0을 넣으면 -1
    loss_init_vel = torch.mean((v0_pred - (-1.0))**2)

    loss = loss_col + beta * (loss_init_pos + loss_init_vel)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        pbar.set_postfix({'loss': f'{loss.item():.6f}', 'best_ls': f'{ls:.6f}'})

    if epoch % 1000==0 and loss.item() < ls:
        ls = loss.item()
        torch.save(model.state_dict(), './params/ocs.pt')

print(f"mse: {ls}")
print(f"to={y0_pred.item()}")

# 학습 후 검증
t = torch.linspace(0, 1, 100).to(device).reshape(-1, 1)
y_sol = torch.exp(-t) * torch.cos(w* t)

model.load_state_dict(torch.load('./params/ocs.pt', map_location=device))
model.eval()

with torch.no_grad():
    y_pred = model(t)

plt.figure(figsize=(8,4))
plt.plot(t.cpu().numpy(), y_sol.cpu().numpy(), "--", color="black")
plt.plot(t.cpu().numpy(), y_pred.detach().cpu().numpy(), color='red', linewidth=1)
plt.title(f"PINN Harmonic Oscillator: $m={params['m']}, \mu={params['mu']}, \omega={w:.2f}, y_0={y_init}$")
plt.xlabel("Time (t)")
plt.ylabel("Displacement (y)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
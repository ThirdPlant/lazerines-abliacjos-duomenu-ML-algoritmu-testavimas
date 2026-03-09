import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


torch.manual_seed(0)


# Training data
x = torch.linspace(-10, 10, 400)
y_true = torch.sin(x) * torch.exp(-0.1 * x)


# Deep net that outputs one global [a, b, c] vector
net = nn.Sequential(
    nn.Linear(1, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 32),
    nn.Tanh(),
    nn.Linear(32, 8),
)
z = torch.ones(1, 1)  # Single dummy input

optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()

def polynomials(x, coeffs):
    return sum(coeffs[i]*x**i for i in range(len(coeffs)))

plt.ion()
fig, ax = plt.subplots()
line_true, = ax.plot(x, y_true, label="true")
line_fit,  = ax.plot(x, y_true, label="fit")   # placeholder
ax.set_ylim(-10, 10)
ax.legend()





steps = 100000
for step in range(steps):
    coeffs = net(z).squeeze(0)  # [3]
    #a, b, c = coeffs
    y_pred = polynomials(x, coeffs)
    loss = loss_fn(y_pred, y_true)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"step={step:4d} loss={loss.item():.6f}")
        with torch.inference_mode():
            y_fit = polynomials(x, coeffs)
        line_fit.set_ydata(y_fit)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()


with torch.inference_mode():
    coeffs = net(z).squeeze(0)
    y_fit = polynomials(x, coeffs)



plt.ioff()
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

x = torch.arange(0, 10, 0.1)
y = torch.sin(x)

x_use = x.unsqueeze(1)
y_use = y.unsqueeze(1)

num_graphs = 2

plt.ion()
fig, axs = plt.subplots(1, num_graphs)
line_fits = []
for i in range(num_graphs):
    axs[i].plot(x, y, label=f"line{i}")
    line_fit, = axs[i].plot(x, y, label=f"line{i}_fit") # placeholderis
    line_fits.append(line_fit)
    axs[i].legend()


def make_model():
    return nn.Sequential(
        nn.Linear(1, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

loss_fn = nn.MSELoss()
models = [make_model() for _ in range(num_graphs)]

optimizers = [
    torch.optim.Adam(models[k].parameters(), lr = 0.01)
    for k in range(num_graphs)
]

for t in range(3000):
    for k in range(num_graphs):
        model = models[k]
        y_pred = model(x_use)
        loss = loss_fn(y_pred, y_use)
        optimizer = optimizers[k]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if t % 100 == 0:
        with torch.inference_mode():
            for d in range(num_graphs):
                y_fit = models[d](x_use).squeeze(1)
                line_fits[d].set_ydata(y_fit)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()


with torch.inference_mode():

    x_extrapolate = torch.arange(-5, 15, 0.1).unsqueeze(1)
    extrapolate = models[num_graphs-1](x_extrapolate).squeeze(1)
    line_fits[num_graphs-1].set_data(x_extrapolate.squeeze(1), extrapolate)
    axs[num_graphs-1].plot(x_extrapolate, torch.sin(x_extrapolate), label="true sin graph")
    axs[num_graphs-1].legend()

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

plt.ioff()
plt.show()
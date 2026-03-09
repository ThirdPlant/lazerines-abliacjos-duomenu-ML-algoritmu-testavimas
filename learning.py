import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Real-valued domain for this formula is x > 0 (sqrt and division by x**5)
x = torch.linspace(0.1, 10, 1000)
y = 4 / (x**3 + (torch.sqrt(x**3) / (2 * x**5)))

x_in = x.unsqueeze(1)          # (2000, 1)
y_true = y.unsqueeze(1)        # (2000, 1)

def make_model():
    return nn.Sequential(
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )

loss_fn = nn.MSELoss()


plt.ion()
fig, ax = plt.subplots()
line_true, = ax.plot(x, y, label="true")
line_fit,  = ax.plot(x, y, label="fit")   # placeholder
ax.legend()
ax.set_ylim(0, 10)

update_every = 100


for i in range(5):
    model=make_model()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=(0.9+0.01*i))
    ax.set_title(f"run {i}, lr={0.02}, momentum={0.9+0.02*i}")

    for t in range(5000):
        y_pred = model(x_in)
        loss = loss_fn(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % update_every == 0:
            with torch.inference_mode():
                y_fit = model(x_in).squeeze(1)
            line_fit.set_ydata(y_fit)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    

    
plt.ioff()

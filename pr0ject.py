
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.arima.model import ARIMA
def generate_synthetic_timeseries(n_samples=20000, seq_len=120, forecast_len=30):
    t = np.arange(n_samples + seq_len + forecast_len)

    # Nonlinear trend
    trend = 0.0005 * (t ** 2)

    # Multi-seasonality
    daily = 2 * np.sin(2 * np.pi * t / 24)
    weekly = 5 * np.sin(2 * np.pi * t / 168)
    yearly = 10 * np.sin(2 * np.pi * t / 8760)

    noise = np.random.normal(scale=2, size=len(t))

    series = trend + daily + weekly + yearly + noise

    X, y = [], []
    for i in range(n_samples):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len:i + seq_len + forecast_len])

    return np.array(X), np.array(y), series


# Generate dataset
SEQ_LEN = 120
FORECAST_LEN = 30

X, y, full_series = generate_synthetic_timeseries(seq_len=SEQ_LEN, forecast_len=FORECAST_LEN)
print("Dataset:", X.shape, y.shape)
class NBeatsBlock(nn.Module):
    def init(self, input_size, theta_size, n_layers=4, layer_size=256):
        super().init()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_size if i == 0 else layer_size, layer_size))
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        self.theta = nn.Linear(layer_size, theta_size)

    def forward(self, x):
        return self.theta(self.fc(x))


class TrendGenerator(nn.Module):
    def init(self, degree, backcast_len, forecast_len):
        super().init()
        self.degree = degree
        self.powers = torch.arange(degree + 1).float()
        self.backcast_t = torch.linspace(-1, 0, backcast_len)
        self.forecast_t = torch.linspace(0, 1, forecast_len)

    def forward(self, theta):
        theta_back = theta[:, :self.degree+1]
        theta_for = theta[:, self.degree+1:]

        backcast = (theta_back.unsqueeze(-1) * (self.backcast_t ** self.powers[:, None])).sum(1)
        forecast = (theta_for.unsqueeze(-1) * (self.forecast_t ** self.powers[:, None])).sum(1)

        return backcast, forecast


class SeasonalityGenerator(nn.Module):
    def init(self, harmonics, backcast_len, forecast_len):
        super().init()
        self.h = harmonics
        f = torch.arange(1, harmonics + 1).float()

        self.bc_cos = torch.cos(2 * np.pi * f[:, None] * torch.linspace(0, 1, backcast_len))
        self.bc_sin = torch.sin(2 * np.pi * f[:, None] * torch.linspace(0, 1, backcast_len))

        self.fc_cos = torch.cos(2 * np.pi * f[:, None] * torch.linspace(0, 1, forecast_len))
        self.fc_sin = torch.sin(2 * np.pi * f[:, None] * torch.linspace(0, 1, forecast_len))

    def forward(self, theta):
        cos_part = theta[:, :self.h]
        sin_part = theta[:, self.h:]

        backcast = cos_part @ self.bc_cos + sin_part @ self.bc_sin
        forecast = cos_part @ self.fc_cos + sin_part @ self.fc_sin

        return backcast, forecast


class NBeats(nn.Module):
    def init(self, input_size, forecast_len):
        super().init()

        self.blocks = nn.ModuleList()

        stack_types = ["generic", "trend", "seasonality"]
        n_blocks = 3

        for stack in stack_types:
            for _ in range(n_blocks):
if stack == "generic":
                    theta_size = input_size + forecast_len
                    gen = None
                elif stack == "trend":
                    degree = 3
                    theta_size = 2 * (degree + 1)
                    gen = TrendGenerator(degree, input_size, forecast_len)
                else:
                    harmonics = 5
                    theta_size = 2 * harmonics
                    gen = SeasonalityGenerator(harmonics, input_size, forecast_len)

                block = NBeatsBlock(input_size=input_size, theta_size=theta_size)
                self.blocks.append((block, gen))

        self.forecast_len = forecast_len
        self.input_size = input_size

    def forward(self, x):
        residual = x
        batch = x.shape[0]
        forecast = torch.zeros(batch, self.forecast_len).to(x.device)

        for block, gen in self.blocks:
            theta = block(residual)

            if gen is None:
                backcast = theta[:, :self.input_size]
                fcast = theta[:, self.input_size:]
            else:
                backcast, fcast = gen(theta)

            residual = residual - backcast
            forecast += fcast

        return forecast
device = "cuda" if torch.cuda.is_available() else "cpu"

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=True)

model = NBeats(input_size=SEQ_LEN, forecast_len=FORECAST_LEN).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

print("Training N-BEATS...")
for epoch in range(5):
    losses = []
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}: Loss = {np.mean(losses):.4f}")
model.eval()
test_input = X_tensor[:1]
pred = model(test_input).detach().cpu().numpy().flatten()

plt.figure(figsize=(12, 5))
plt.plot(range(SEQ_LEN), test_input.cpu().numpy().flatten(), label="Input Sequence")
plt.plot(range(SEQ_LEN, SEQ_LEN + FORECAST_LEN), pred, label="N-BEATS Forecast")
plt.legend()
plt.title("N-BEATS Forecast")
plt.show()

print("Training ARIMA baseline (may take time)...")
arima_model = ARIMA(full_series[:-FORECAST_LEN], order=(5,1,2)).fit()
arima_forecast = arima_model.forecast(steps=FORECAST_LEN)

plt.figure(figsize=(12,5))
plt.plot(arima_forecast, label="ARIMA Forecast")
plt.plot(pred, label="N-BEATS Forecast")
plt.title("ARIMA vs N-BEATS Comparison")
plt.legend()
plt.show()

print("Done.")

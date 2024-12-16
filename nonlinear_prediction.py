import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Definice datasetu
class NonLinearDataset(Dataset):
    def __init__(self):
        self.x = torch.linspace(-10, 10, 200).unsqueeze(1)  # nezavisla promenna
        self.y = 4*self.x**3 +self.x**2 + 2 * self.x + 1 + 150 * torch.randn_like(self.x)  # zavisla promenna, polynom + sum

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Definice modelu nn
class NonLinearModel(nn.Module):
    def __init__(self):
        super(NonLinearModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, 1)
        )

    def forward(self, x):
        return self.network(x)

# Rozdělení na tréningový a testovací sety
full_dataset = NonLinearDataset()
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoaders pro tréningový a testovací set
dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Inicializace modelu, loss funkcie a optimizéra
model = NonLinearModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Tréningová slučka s learning rate decay
num_epochs = 200
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Vytvoření proměnných pro ztráty a data pro plotování
losses = []
x_vals = full_dataset.x.numpy()
y_true = full_dataset.y.numpy()

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_x, batch_y in dataloader:
        # Predikcia
        outputs = model(batch_x)
        
        # Výpočet chyby
        loss = criterion(outputs, batch_y)
        epoch_loss += loss.item()
        
        # Backpropagation a optimalizácia
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    losses.append(epoch_loss / len(dataloader))

    # Vizualizace každých 20 epoch
    if (epoch + 1) % 20 == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            y_pred = model(torch.tensor(x_vals, dtype=torch.float32)).numpy()
        plt.figure(figsize=(10, 5))
        plt.scatter(x_vals, y_true, label="Skutečná data", alpha=0.6)
        plt.plot(x_vals, y_pred, color="red", label="Predikce")
        plt.title(f"Epoch {epoch + 1}")
        plt.legend()
        plt.pause(2)  # Zobrazí graf na 5 sekundy
        plt.close()

    # Vypsaní loss
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}, LR: {scheduler.get_last_lr()[0]:.5f}")
        
# Vizualizace ztráty
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), losses, label="Loss")
plt.title("Průběh MSELoss v rámci epoch")
plt.xlabel("Epochy")
plt.ylabel("MSELoss")
plt.legend()
plt.pause(5)  # Zobrazí graf na 3 sekundy
plt.close()

# Vyhodnocení na test. sete
model.eval()
test_losses = []
test_x_vals = []
test_y_true = []
test_y_pred = []

with torch.no_grad():
    for batch_x, batch_y in test_dataloader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        test_losses.append(loss.item())
        test_x_vals.append(batch_x.numpy())
        test_y_true.append(batch_y.numpy())
        test_y_pred.append(outputs.numpy())

average_test_loss = sum(test_losses) / len(test_losses)
print(f"Průměrná MSELoss: {average_test_loss:.4f}")

test_x_vals = torch.cat([torch.tensor(arr) for arr in test_x_vals]).numpy()
test_y_true = torch.cat([torch.tensor(arr) for arr in test_y_true]).numpy()
test_y_pred = torch.cat([torch.tensor(arr) for arr in test_y_pred]).numpy()

plt.figure(figsize=(10, 5))
plt.scatter(test_x_vals, test_y_true, label="Skutočná data", alpha=0.6)
plt.scatter(test_x_vals, test_y_pred, color="red", label="Predikce", alpha=0.6)
plt.title("Predikce na testovacím sete")
plt.legend()
plt.show()

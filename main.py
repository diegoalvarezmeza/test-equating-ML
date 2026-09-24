"""VAE ordinal: el decoder predice una puntuación continua y K-1 umbrales gamma
(por defecto fijos, sin aprender) definen los cortes entre categorías -- modelo
logístico acumulado (proportional odds). El encoder soporta distintas formas de
representar la entrada (input_mode="onehot"/"termometro"/"embedding"), pero acá
solo corremos la versión one-hot.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score

import utils

utils.set_seed(42)
device = utils.get_device()
print(f"Using device: {device}")

K = 15
N = 5000

# 1. simulamos una distribución discreta

true_pmf = utils.pmf_gaussiana(K, mu=10, sigma=3)
scores = utils.simulate_group(true_pmf, N)
emp_pmf = utils.empirical_pmf(scores, K)

loader, tensor = utils.make_loader(scores, K)


# 2. VAE ordinal: encoder -> (mu, logvar) -> z -> decoder -> puntuación continua s

class OrdinalVAE(nn.Module):
    def __init__(self, k, hidden=128, latent_dim=1, input_mode="onehot", threshold_mode="aprendido"):
        super().__init__()
        self.k = k
        self.latent_dim = latent_dim
        self.input_mode = input_mode
        self.threshold_mode = threshold_mode

        if input_mode == "onehot":
            input_dim = k
        elif input_mode == "termometro":
            input_dim = k - 1
        elif input_mode == "embedding":
            input_dim = 1
            self.input_base = nn.Parameter(torch.zeros(1))
            self.input_deltas = nn.Parameter(torch.zeros(k - 1))
        else:
            raise ValueError(f"input_mode desconocido: {input_mode}")

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.mu_layer = nn.Linear(hidden, latent_dim)
        self.logvar_layer = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

        gamma_base = torch.zeros(1)
        gamma_deltas = torch.zeros(k - 2)
        if threshold_mode == "aprendido":
            self.gamma_base = nn.Parameter(gamma_base)
            self.gamma_deltas = nn.Parameter(gamma_deltas)
        elif threshold_mode == "fijo":
            self.register_buffer("gamma_base", gamma_base)
            self.register_buffer("gamma_deltas", gamma_deltas)
        else:
            raise ValueError(f"threshold_mode desconocido: {threshold_mode}")

    def thresholds(self):
        """gamma_1 < ... < gamma_{k-1}, crecientes porque softplus siempre es positivo."""
        deltas = nn.functional.softplus(self.gamma_deltas)
        return torch.cat([self.gamma_base, self.gamma_base + torch.cumsum(deltas, dim=0)])

    def category_embedding(self):
        """Solo input_mode='embedding': k posiciones e_1 < ... < e_K, una por categoría."""
        deltas = nn.functional.softplus(self.input_deltas)
        return torch.cat([self.input_base, self.input_base + torch.cumsum(deltas, dim=0)])

    def prepare_input(self, x_onehot):
        if self.input_mode == "onehot":
            return x_onehot
        if self.input_mode == "termometro":
            cum = torch.cumsum(x_onehot, dim=-1)
            return 1.0 - cum[:, :-1]
        emb = self.category_embedding()
        return (x_onehot * emb[None, :]).sum(dim=1, keepdim=True)

    def category_probs(self, s):
        gammas = self.thresholds()
        cum = torch.sigmoid(gammas[None, :] - s[:, None])
        ones = torch.ones(s.shape[0], 1, device=s.device)
        zeros = torch.zeros(s.shape[0], 1, device=s.device)
        cum_full = torch.cat([zeros, cum, ones], dim=1)
        return (cum_full[:, 1:] - cum_full[:, :-1]).clamp_min(1e-8)

    def encode(self, x_onehot):
        h = self.encoder(self.prepare_input(x_onehot))
        return self.mu_layer(h), self.logvar_layer(h)

    def forward(self, x_onehot):
        mu, logvar = self.encode(x_onehot)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        s = self.decoder(z).squeeze(-1)
        return self.category_probs(s), mu, logvar

    @torch.no_grad()
    def estimate_pmf(self, n_samples=10000):
        z = torch.randn(n_samples, self.latent_dim, device=next(self.parameters()).device)
        s = self.decoder(z).squeeze(-1)
        return self.category_probs(s).mean(dim=0).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, x_onehot):
        mu, _ = self.encode(x_onehot)
        s = self.decoder(mu).squeeze(-1)
        return self.category_probs(s).argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def latent_vectors(self, x_onehot):
        mu, _ = self.encode(x_onehot)
        return mu.cpu().numpy()


def vae_loss(x, probs, mu, logvar, beta=1.0):
    labels = x.argmax(dim=1)
    log_probs = torch.log(probs.gather(1, labels.unsqueeze(1)).squeeze(1))
    recon = -log_probs.sum()
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl


def train_vae(vae, loader, n_total, epochs=300, warmup_epochs=100, beta_max=0.5):
    """beta sube de 0 a beta_max en warmup_epochs, para evitar el colapso posterior."""
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        beta = beta_max * min(1.0, epoch / warmup_epochs)
        vae.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            probs, mu, logvar = vae(batch)
            loss = vae_loss(batch, probs, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"  epoch {epoch}/{epochs}  beta={beta:.2f}  loss/muestra={total_loss / n_total:.4f}")


# 3. entrenamos el VAE ordinal

vae = OrdinalVAE(k=K, input_mode="onehot").to(device)
train_vae(vae, loader, N, epochs=300, warmup_epochs=100)
vae.eval()

gammas = vae.thresholds().detach().cpu().numpy()
pred = vae.reconstruct(tensor.to(device))
pmf_vae = vae.estimate_pmf()
accuracy = accuracy_score(scores, pred)

print(f"\naccuracy={accuracy:.3f}  "
      f"TV={utils.tv_distance(pmf_vae, true_pmf):.4f}  "
      f"Wasserstein={utils.wasserstein_pmf(pmf_vae, true_pmf):.4f}")
print(f"umbrales gamma: {np.round(gammas, 3).tolist()}")
print(f"TV/Wasserstein empírica de referencia: "
      f"TV={utils.tv_distance(emp_pmf, true_pmf):.4f}  "
      f"Wasserstein={utils.wasserstein_pmf(emp_pmf, true_pmf):.4f}")


# 4. gráficos: PMF, matriz de confusión y densidad latente

x = np.arange(1, K + 1)
colors = plt.cm.viridis(np.linspace(0, 1, K))
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

axes[0].bar(x - 0.25, true_pmf, width=0.25, label="Verdadera")
axes[0].bar(x, emp_pmf, width=0.25, label="Empírica")
axes[0].bar(x + 0.25, pmf_vae, width=0.25, label="VAE")
axes[0].set_title("PMF"); axes[0].set_xlabel("Valor"); axes[0].set_ylabel("Probabilidad")
axes[0].legend(fontsize=7)

cm = utils.matriz_confusion(scores, pred, K)
axes[1].imshow(cm, cmap="Blues")
for i in range(K):
    for j in range(K):
        axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
axes[1].set_title(f"Reconstrucción — accuracy={accuracy:.3f}")
axes[1].set_xlabel("Predicho"); axes[1].set_ylabel("Verdadero")

with torch.no_grad():
    z_prior = torch.randn(10000, vae.latent_dim, device=device)
    s_samples = vae.decoder(z_prior).squeeze(-1).cpu().numpy()

# continuización: reescalamos s al eje de "Valor" (1..K) anclando cada umbral
# gamma_j al borde real entre categorías j y j+1 (j + 0.5), para que la densidad
# quede en la misma escala que la PMF discreta del panel 0.
bordes_valor = np.arange(1, K) + 0.5
s_a_valor = interp1d(gammas, bordes_valor, kind="linear", fill_value="extrapolate")
value_samples = s_a_valor(s_samples)

kde = gaussian_kde(value_samples)
v_grid = np.linspace(0.5, K + 0.5, 300)
densidad_v = kde(v_grid)
categoria_por_v = np.clip(np.searchsorted(bordes_valor, v_grid), 0, K - 1)

axes[2].plot(v_grid, densidad_v, color="black", lw=0.8)
for j in range(K):
    mask = categoria_por_v == j
    if mask.any():
        axes[2].fill_between(v_grid[mask], densidad_v[mask], color=colors[j])
for b in bordes_valor:
    axes[2].axvline(b, color="white", lw=0.6, alpha=0.7)
axes[2].set_title("Densidad continuizada"); axes[2].set_xlabel("Valor"); axes[2].set_ylabel("Densidad")
axes[2].set_xlim(x.min() - 0.5, x.max() + 0.5)

plt.tight_layout()
plt.show()

"""VAE ordinal: el decoder predice una puntuación continua y K-1 umbrales gamma (aprendidos,
siempre ordenados) definen los cortes entre categorías -- modelo logístico acumulado
(proportional odds). Compara tres formas de representar la entrada al encoder: one-hot
nominal, un termómetro fijo, y un embedding ordenado aprendido por categoría.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, k, hidden=128, latent_dim=1, input_mode="onehot"):
        super().__init__()
        self.k = k
        self.latent_dim = latent_dim
        self.input_mode = input_mode

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
        self.gamma_base = nn.Parameter(torch.zeros(1))
        self.gamma_deltas = nn.Parameter(torch.zeros(k - 2))

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


# 3. entrenamos las 3 variantes de entrada y comparamos

resultados = {}
for modo in ["onehot", "termometro", "embedding"]:
    print(f"\n=== Entrenando VAE ordinal — input_mode='{modo}' ===")
    vae = OrdinalVAE(k=K, input_mode=modo).to(device)
    train_vae(vae, loader, N, epochs=300, warmup_epochs=100)
    vae.eval()

    gammas = vae.thresholds().detach().cpu().numpy()
    pred = vae.reconstruct(tensor.to(device))
    pmf_vae = vae.estimate_pmf()

    resultados[modo] = {
        "vae": vae,
        "gammas": gammas,
        "pred": pred,
        "pmf_vae": pmf_vae,
        "accuracy": accuracy_score(scores, pred),
        "tv": utils.tv_distance(pmf_vae, true_pmf),
        "wasserstein": utils.wasserstein_pmf(pmf_vae, true_pmf),
    }
    print(f"  accuracy={resultados[modo]['accuracy']:.3f}  "
          f"TV={resultados[modo]['tv']:.4f}  Wasserstein={resultados[modo]['wasserstein']:.4f}")

print("\n=== Comparación de estrategias de codificación de entrada ===")
print(f'{"input_mode":12} {"Accuracy":>9} {"TV":>8} {"Wasserstein":>12}')
for modo, r in resultados.items():
    print(f'{modo:12} {r["accuracy"]:>9.3f} {r["tv"]:>8.4f} {r["wasserstein"]:>12.4f}')

print(f"\nTV/Wasserstein empírica de referencia: "
      f"TV={utils.tv_distance(emp_pmf, true_pmf):.4f}  "
      f"Wasserstein={utils.wasserstein_pmf(emp_pmf, true_pmf):.4f}")


# 4. gráficos: PMF, matriz de confusión y densidad latente, por cada input_mode

x = np.arange(1, K + 1)
colors = plt.cm.viridis(np.linspace(0, 1, K))
fig, axes = plt.subplots(3, 3, figsize=(18, 12))

for col, modo in enumerate(["onehot", "termometro", "embedding"]):
    r = resultados[modo]

    ax_pmf = axes[0, col]
    ax_pmf.bar(x - 0.25, true_pmf, width=0.25, label="Verdadera")
    ax_pmf.bar(x, emp_pmf, width=0.25, label="Empírica")
    ax_pmf.bar(x + 0.25, r["pmf_vae"], width=0.25, label="VAE")
    ax_pmf.set_title(f"PMF — input_mode='{modo}'")
    ax_pmf.set_xlabel("Valor"); ax_pmf.set_ylabel("Probabilidad"); ax_pmf.legend(fontsize=7)

    cm = utils.matriz_confusion(scores, r["pred"], K)
    ax_cm = axes[1, col]
    ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_title(f"Reconstrucción — accuracy={r['accuracy']:.3f}")
    ax_cm.set_xlabel("Predicho"); ax_cm.set_ylabel("Verdadero")

    vae_m, gammas_m = r["vae"], r["gammas"]
    with torch.no_grad():
        z_prior = torch.randn(10000, vae_m.latent_dim, device=device)
        s_samples = vae_m.decoder(z_prior).squeeze(-1).cpu().numpy()

    kde = gaussian_kde(s_samples)
    s_grid = np.linspace(s_samples.min(), s_samples.max(), 300)
    densidad_s = kde(s_grid)
    categoria_por_s = np.searchsorted(gammas_m, s_grid)

    ax_dens = axes[2, col]
    ax_dens.plot(s_grid, densidad_s, color="black", lw=0.8)
    for j in range(K):
        mask = categoria_por_s == j
        if mask.any():
            ax_dens.fill_between(s_grid[mask], densidad_s[mask], color=colors[j])
    for g in gammas_m:
        ax_dens.axvline(g, color="white", lw=0.6, alpha=0.7)
    ax_dens.set_title(f"Densidad de s — input_mode='{modo}'")
    ax_dens.set_xlabel("Puntuación continua s"); ax_dens.set_ylabel("Densidad")

plt.tight_layout()
plt.show()

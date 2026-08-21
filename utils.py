import numpy as np
import torch
from scipy.spatial.distance import cityblock
from scipy.stats import wasserstein_distance, norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset


# reproducibilidad y device

def set_seed(seed=42):
    """Fija la semilla de numpy y torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    """'cuda' si hay GPU disponible, si no 'cpu'."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PMFs de ejemplo y simulación de datos

def pmf_desde_pesos(pesos):
    """Normaliza un arreglo de pesos no negativos a una PMF válida."""
    pesos = np.asarray(pesos, dtype=float)
    return pesos / pesos.sum()


def pmf_gaussiana(k, mu, sigma):
    """PMF unimodal: gaussiana discretizada sobre {0,...,k-1}, normalizada."""
    x = np.arange(k)
    pmf = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return pmf / pmf.sum()


def pmf_bimodal(k, mu1, sigma1, mu2, sigma2, peso1=0.5):
    """Mezcla de dos gaussianas discretizadas sobre {0,...,k-1}, normalizada a PMF."""
    x = np.arange(k)
    comp1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
    comp2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
    pmf = peso1 * comp1 + (1 - peso1) * comp2
    return pmf / pmf.sum()


def pmf_prueba_A_ejemplo():
    """Prueba A (K=10): asimétrica negativa, mayoría nota media-alta."""
    return pmf_desde_pesos([0.02, 0.04, 0.06, 0.09, 0.13, 0.17, 0.19, 0.15, 0.10, 0.05])


def pmf_prueba_B_ejemplo(k=100):
    """Prueba B (K=100 por defecto): mezcla bimodal, prueba más difícil."""
    return pmf_bimodal(k, mu1=30, sigma1=12, mu2=68, sigma2=15, peso1=0.55)


def simulate_group(pmf, n_samples):
    """Simula puntajes (índices 0-based) para un grupo dado una PMF y un tamaño de muestra."""
    return np.random.choice(len(pmf), size=n_samples, p=pmf)


def empirical_pmf(scores, k):
    """PMF empírica (histograma normalizado) de puntajes discretos en {0,...,k-1}."""
    return np.bincount(scores, minlength=k) / len(scores)


def simulate_two_groups(pmf_A, pmf_B, n_A, n_B):
    """Simula dos grupos independientes (cada uno rinde solo una prueba) y sus PMFs empíricas."""
    scores_A = simulate_group(pmf_A, n_A)
    scores_B = simulate_group(pmf_B, n_B)
    emp_pmf_A = empirical_pmf(scores_A, len(pmf_A))
    emp_pmf_B = empirical_pmf(scores_B, len(pmf_B))
    return scores_A, scores_B, emp_pmf_A, emp_pmf_B


# preprocesamiento

def to_onehot(indices, k):
    """One-hot (N, k) vía sklearn.OneHotEncoder."""
    encoder = OneHotEncoder(categories=[np.arange(k)], sparse_output=False, dtype=np.float32)
    return encoder.fit_transform(np.asarray(indices).reshape(-1, 1))


def make_loader(indices, k, batch_size=256, shuffle=True):
    """DataLoader de tensores one-hot a partir de índices de puntaje."""
    tensor = torch.tensor(to_onehot(indices, k))
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle)
    return loader, tensor


# equiparación por percentiles (empírica / VAE)

def equiparar(pmf_A, pmf_B):
    """Tabla de conversión 0-indexed: para cada categoría de A, la más cercana en percentil en B."""
    cdf_A = np.cumsum(pmf_A)
    cdf_B = np.cumsum(pmf_B)
    return np.argmin(np.abs(cdf_B[None, :] - cdf_A[:, None]), axis=1)


# equiparación por kernel (von Davier, Holland & Thayer) -- baseline no-VAE

def _continuizar(pmf, n_obs, support):
    """Media, ancho de banda (regla de Silverman) y factor de corrección de varianza."""
    mu = np.average(support, weights=pmf)
    sigma = np.sqrt(np.average((support - mu) ** 2, weights=pmf))
    h = sigma * (4 / (3 * n_obs)) ** 0.2
    a = np.sqrt(sigma ** 2 / (sigma ** 2 + h ** 2))
    return mu, h, a


def gaussian_kernel_pmf(pmf, n_obs, support=None):
    """PMF suavizada con un kernel gaussiano, evaluada de nuevo en el soporte entero."""
    support = np.arange(len(pmf)) if support is None else np.asarray(support, dtype=float)
    mu, h, a = _continuizar(pmf, n_obs, support)
    z = (support[:, None] - a * support[None, :] - (1 - a) * mu) / (a * h)
    densidad = (norm.pdf(z) * pmf[None, :]).sum(axis=1)
    return densidad / densidad.sum()


def gaussian_kernel_cdf(x, pmf, n_obs, support=None):
    """CDF continuizada del mismo kernel, evaluada en los puntos `x` (por ejemplo una grilla fina)."""
    support = np.arange(len(pmf)) if support is None else np.asarray(support, dtype=float)
    mu, h, a = _continuizar(pmf, n_obs, support)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    z = (x[:, None] - a * support[None, :] - (1 - a) * mu) / (a * h)
    return (norm.cdf(z) * pmf[None, :]).sum(axis=1)


def kernel_equiparar(pmf_from, pmf_to, n_obs_from, n_obs_to):
    """Percentil en 'from' -> mismo percentil en la CDF continuizada de 'to'."""
    percentiles = gaussian_kernel_cdf(np.arange(len(pmf_from)), pmf_from, n_obs_from)
    grid = np.linspace(0, len(pmf_to) - 1, 2000)
    cdf_to = gaussian_kernel_cdf(grid, pmf_to, n_obs_to)
    equivalentes = np.interp(percentiles, cdf_to, grid)
    return np.clip(np.round(equivalentes), 0, len(pmf_to) - 1).astype(int)


# métricas

def tv_distance(p, q):
    """Distancia de variación total entre dos PMFs (mitad de la distancia L1 / cityblock)."""
    return 0.5 * cityblock(p, q)


def wasserstein_pmf(p, q, support=None):
    """Distancia de Wasserstein-1 entre dos PMFs discretas sobre el mismo soporte."""
    if support is None:
        support = np.arange(len(p))
    return wasserstein_distance(support, support, p, q)


def is_monotonic(conversion):
    """True si la tabla de conversión es monótona no decreciente."""
    return bool(np.all(np.diff(np.asarray(conversion)) >= 0))


def decilizar(scores, k, n_grupos=10):
    """Agrupa puntajes discretos {0,...,k-1} en n_grupos bins de igual tamaño (deciles por defecto)."""
    tam = k / n_grupos
    grupos = np.floor(np.asarray(scores) / tam).astype(int)
    return np.clip(grupos, 0, n_grupos - 1)


def matriz_confusion(true_labels, pred_labels, k):
    """Matriz de confusión k x k (etiquetas 0..k-1) entre categoría verdadera y reconstruida."""
    return confusion_matrix(true_labels, pred_labels, labels=np.arange(k))

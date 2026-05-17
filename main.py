"""
ismail ambarkütük

cmpe 442 programming assignment 3
image segmentation with k means and gaussian mixture model from scratch

run
python main.py
python main.py --image_dir images --out_dir outputs --k_values 2 4 6 8 --runs 3

notes
this code uses relative paths by default
no ready to use ml implementation is used
numpy is used for matrix operations
pil and matplotlib are used for image io and visual output
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple, Dict, List

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# utility funcs

def load_image(path: Path, max_side: int | None = 350) -> np.ndarray:
    """load rgb image as float array and resize if needed"""
    img = Image.open(path).convert("RGB")
    if max_side is not None and max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (int(round(img.size[0] * scale)), int(round(img.size[1] * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float64) / 255.0


def make_features(image: np.ndarray, spatial_weight: float = 0.15) -> np.ndarray:
    ''"""
    convert image to pixel feature matrix
    rgb values are main features and x y positions are added for more spatialy stable segments
    """
    h, w, _ = image.shape
    rgb = image.reshape(-1, 3)

    if spatial_weight <= 0:
        return rgb

    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.column_stack((xx.reshape(-1) / max(w - 1, 1), yy.reshape(-1) / max(h - 1, 1)))
    return np.column_stack((rgb, spatial_weight * coords))


def labels_to_segmented_image(labels: np.ndarray, image: np.ndarray, k: int) -> np.ndarray:
    """paint every segment with mean rgb color of its pixels"""
    flat_rgb = image.reshape(-1, 3)
    out = np.zeros_like(flat_rgb)
    global_mean = flat_rgb.mean(axis=0)

    for cluster_id in range(k):
        mask = labels == cluster_id
        if np.any(mask):
            out[mask] = flat_rgb[mask].mean(axis=0)
        else:
            out[mask] = global_mean

    return out.reshape(image.shape)


def save_image(image: np.ndarray, path: Path) -> None:
    """save float rgb image"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)


# k means from scrach

def kmeans_plus_plus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k means plus plus init with numpy"""
    n_samples = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float64)

    first_idx = rng.integers(n_samples)
    centers[0] = x[first_idx]

    closest_dist_sq = np.sum((x - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = closest_dist_sq.sum()
        if total <= 1e-15:
            centers[c] = x[rng.integers(n_samples)]
            continue
        probabilities = closest_dist_sq / total
        idx = rng.choice(n_samples, p=probabilities)
        centers[c] = x[idx]
        dist_sq = np.sum((x - centers[c]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)

    return centers


def run_kmeans(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """run one k means fit and return labels centers and inertia"""
    centers = kmeans_plus_plus_init(x, k, rng)
    labels = np.zeros(x.shape[0], dtype=np.int64)

    for _ in range(max_iter):
        # squared euclidean distance from every pixel to every center
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for cluster_id in range(k):
            mask = new_labels == cluster_id
            if np.any(mask):
                new_centers[cluster_id] = x[mask].mean(axis=0)
            else:
                # init empty cluster again with a random pixel feautre
                new_centers[cluster_id] = x[rng.integers(x.shape[0])]

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        labels = new_labels
        if shift < tol:
            break

    inertia = float(np.sum((x - centers[labels]) ** 2))
    return labels, centers, inertia


def best_kmeans(
    x: np.ndarray,
    k: int,
    runs: int,
    seed: int,
    max_iter: int,
) -> Dict[str, np.ndarray | float]:
    """run k means more than once and keep the lowest inertia result"""
    best = None
    for run in range(runs):
        rng = np.random.default_rng(seed + 1000 * run + 17 * k)
        labels, centers, inertia = run_kmeans(x, k, rng, max_iter=max_iter)
        if best is None or inertia < best["score"]:
            best = {"labels": labels, "centers": centers, "score": inertia}
    return best


# gmm from scratch with em algortihm

def logsumexp(a: np.ndarray, axis: int = 1, keepdims: bool = False) -> np.ndarray:
    """stable log sum exp without scipy"""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True) + 1e-300)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def diagonal_gaussian_logpdf(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """log density for diagonal covariance gaussians"""
    d = x.shape[1]
    diff_sq = (x[:, None, :] - means[None, :, :]) ** 2
    log_det = np.sum(np.log(variances), axis=1)
    maha = np.sum(diff_sq / variances[None, :, :], axis=2)
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det[None, :] + maha)


def initialize_gmm_with_kmeans(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """use short k means run to init gmm parameters"""
    labels, means, _ = run_kmeans(x, k, rng, max_iter=20)
    n, d = x.shape
    weights = np.zeros(k, dtype=np.float64)
    variances = np.zeros((k, d), dtype=np.float64)
    global_var = np.var(x, axis=0) + 1e-4

    for cluster_id in range(k):
        mask = labels == cluster_id
        count = int(np.sum(mask))
        if count > 1:
            weights[cluster_id] = count / n
            variances[cluster_id] = np.var(x[mask], axis=0) + 1e-4
        else:
            weights[cluster_id] = 1.0 / k
            means[cluster_id] = x[rng.integers(n)]
            variances[cluster_id] = global_var

    weights = weights / weights.sum()
    return weights, means, variances


def run_gmm(
    x: np.ndarray,
    k: int,
    rng: np.random.Generator,
    max_iter: int = 80,
    tol: float = 1e-4,
    reg_covar: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """run one em fit for diagonal covariance gmm"""
    n, d = x.shape
    weights, means, variances = initialize_gmm_with_kmeans(x, k, rng)
    previous_ll = -np.inf

    for _ in range(max_iter):
        # e step calculate responsibilites
        log_prob = diagonal_gaussian_logpdf(x, means, variances) + np.log(weights[None, :] + 1e-300)
        log_norm = logsumexp(log_prob, axis=1, keepdims=True)
        responsibilities = np.exp(log_prob - log_norm)
        ll = float(np.sum(log_norm))

        # m step update weights means and variances
        nk = responsibilities.sum(axis=0) + 1e-12
        weights = nk / n
        means = (responsibilities.T @ x) / nk[:, None]

        for cluster_id in range(k):
            diff = x - means[cluster_id]
            variances[cluster_id] = (responsibilities[:, cluster_id][:, None] * diff * diff).sum(axis=0) / nk[cluster_id]
        variances = np.maximum(variances, reg_covar)

        if abs(ll - previous_ll) < tol * (1.0 + abs(previous_ll)):
            break
        previous_ll = ll

    final_log_prob = diagonal_gaussian_logpdf(x, means, variances) + np.log(weights[None, :] + 1e-300)
    labels = np.argmax(final_log_prob, axis=1)
    final_ll = float(np.sum(logsumexp(final_log_prob, axis=1)))
    return labels, weights, means, variances, final_ll


def best_gmm(
    x: np.ndarray,
    k: int,
    runs: int,
    seed: int,
    max_iter: int,
) -> Dict[str, np.ndarray | float]:
    """run gmm more than once and keep best log likelihood result"""
    best = None
    for run in range(runs):
        rng = np.random.default_rng(seed + 2000 * run + 31 * k)
        labels, weights, means, variances, log_likelihood = run_gmm(x, k, rng, max_iter=max_iter)
        if best is None or log_likelihood > best["score"]:
            best = {
                "labels": labels,
                "weights": weights,
                "means": means,
                "variances": variances,
                "score": log_likelihood,
            }
    return best


# experment runner and visual files

def create_comparison_grid(
    original: np.ndarray,
    k_values: Iterable[int],
    kmeans_maps: Dict[int, np.ndarray],
    gmm_maps: Dict[int, np.ndarray],
    save_path: Path,
) -> None:
    """create one compact visual for original k means and gmm results"""
    k_values = list(k_values)
    n_cols = 1 + len(k_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.4))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(original)
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")

    for col, k in enumerate(k_values, start=1):
        axes[0, col].imshow(kmeans_maps[k])
        axes[0, col].set_title(f"K-Means, K={k}")
        axes[0, col].axis("off")

        axes[1, col].imshow(gmm_maps[k])
        axes[1, col].set_title(f"GMM, K={k}")
        axes[1, col].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> None:
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    segmented_dir = out_dir / "segmentation_maps"
    grid_dir = out_dir / "comparison_grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if not image_paths:
        raise FileNotFoundError(f"No images were found in relative folder: {image_dir}")

    print(f"Found {len(image_paths)} image(s): {[p.name for p in image_paths]}")
    print(f"K values: {args.k_values}")

    metrics_lines: List[str] = ["image,algorithm,k,best_score"]

    for image_path in image_paths:
        print(f"\nProcessing {image_path.name}")
        image = load_image(image_path, max_side=args.max_side)
        features = make_features(image, spatial_weight=args.spatial_weight)

        kmeans_maps: Dict[int, np.ndarray] = {}
        gmm_maps: Dict[int, np.ndarray] = {}

        for k in args.k_values:
            print(f"  K={k}: K-Means ...", end="", flush=True)
            km = best_kmeans(features, k, runs=args.runs, seed=args.seed, max_iter=args.kmeans_iter)
            km_map = labels_to_segmented_image(km["labels"], image, k)
            kmeans_maps[k] = km_map
            save_image(km_map, segmented_dir / f"{image_path.stem}_kmeans_K{k}.png")
            metrics_lines.append(f"{image_path.name},kmeans,{k},{km['score']:.6f}")
            print(f" best inertia={km['score']:.4f}")

            print(f"  K={k}: GMM ...", end="", flush=True)
            gm = best_gmm(features, k, runs=args.runs, seed=args.seed, max_iter=args.gmm_iter)
            gm_map = labels_to_segmented_image(gm["labels"], image, k)
            gmm_maps[k] = gm_map
            save_image(gm_map, segmented_dir / f"{image_path.stem}_gmm_K{k}.png")
            metrics_lines.append(f"{image_path.name},gmm,{k},{gm['score']:.6f}")
            print(f" best log-likelihood={gm['score']:.4f}")

        create_comparison_grid(
            original=image,
            k_values=args.k_values,
            kmeans_maps=kmeans_maps,
            gmm_maps=gmm_maps,
            save_path=grid_dir / f"{image_path.stem}_comparison.png",
        )

    (out_dir / "metrics.csv").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    print(f"\nDone. Results are saved under: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-Means and GMM image segmentation from scratch")
    parser.add_argument("--image_dir", type=str, default="images", help="Relative path to the image folder")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Relative path to save outputs")
    parser.add_argument("--k_values", type=int, nargs="+", default=[2, 4, 6, 8], help="Cluster counts")
    parser.add_argument("--runs", type=int, default=3, help="Number of random initializations per K")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_side", type=int, default=350, help="Resize largest image side to this value; use 0 to disable")
    parser.add_argument("--spatial_weight", type=float, default=0.15, help="Weight of normalized x-y coordinates")
    parser.add_argument("--kmeans_iter", type=int, default=100, help="Maximum K-Means iterations")
    parser.add_argument("--gmm_iter", type=int, default=80, help="Maximum GMM EM iterations")
    args = parser.parse_args()
    if args.max_side <= 0:
        args.max_side = None
    return args


if __name__ == "__main__":
    run_experiment(parse_args())

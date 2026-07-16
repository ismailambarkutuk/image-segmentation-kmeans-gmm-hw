# CMPE 442 HW3 — K-Means and GMM Image Segmentation

## Run

```bash
pip install -r requirements.txt
python main.py
```

The default command expects the `images/` folder to be in the same folder as `main.py`, and saves results into `outputs/`.

## Outputs

- `outputs/segmentation_maps/`: individual segmentation maps for 4 images × 4 K values × 2 algorithms = 32 images
- `outputs/comparison_grids/`: compact grids for easier report preparation
- `outputs/metrics.csv`: best inertia/log-likelihood scores 

## Default K values

`K = 2, 4, 6, 8`

You can change them with:

```bash
python main.py --k_values 3 5 7 9
```

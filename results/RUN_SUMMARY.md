# Retail CLIQUE Run Summary

- Selection objective: `quality`
- Selected params: `xi=16, tau=0.2`
- Selected features: `['frequency', 'return_rate', 'weekend_ratio']`
- Train silhouette: `0.3456`
- Coverage: `77.2%`
- Non-noise clusters: `6`

## Top-5 configs by silhouette

| xi | tau | silhouette | coverage | n_clusters | davies_bouldin |
|---:|----:|-----------:|---------:|-----------:|---------------:|
| 16 | 0.20 | 0.3456 | 0.772 | 6 | 1.0788 |
| 12 | 0.10 | 0.3454 | 0.882 | 8 | 1.3871 |
| 14 | 0.20 | 0.3432 | 0.783 | 6 | 1.0772 |
| 10 | 0.20 | 0.3374 | 0.826 | 7 | 1.0279 |
| 10 | 0.18 | 0.3374 | 0.826 | 7 | 1.0279 |

Pareto chart: `D:\clique_customer_app\results\figures\pareto_frontier.png`
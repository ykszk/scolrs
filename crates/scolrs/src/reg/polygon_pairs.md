# Registration with Per-Pair Rigid Transforms and One Global Scale

## Problem Formulation

### Data

Let $K = 7$ be the number of polygon pairs. The $k$-th polygon pair has $N_k$ vertices
(which may differ across pairs). The input is a set of point correspondences

$$
\bigl\{ \mathbf{P}_i^{(k)},\; \mathbf{Q}_i^{(k)} \bigr\}, \qquad k = 1,\ldots,K,\quad i = 1,\ldots,N_k,
$$

where $\mathbf{P}_i^{(k)}, \mathbf{Q}_i^{(k)} \in \mathbb{R}^2$ are the $i$-th vertices of the $k$-th source and target polygon, respectively. The total number of point correspondences is $N_{\text{tot}} = \sum_{k=1}^K N_k$.

### Unknowns

| Symbol | Meaning | DoF |
|--------|---------|-----|
| $\theta_k \in \mathbb{R}$ | Rotation angle for pair $k$ | $K$ |
| $\mathbf{t}_k \in \mathbb{R}^2$ | Translation for pair $k$ | $2K$ |
| $s \in \mathbb{R}_{+}$ | Global scale (shared across all pairs) | $1$ |

**Total: $K \times 3 + 1 = 22$ degrees of freedom** (independent of the $N_k$).

### Transformation Model

Each source point is mapped by a scaled rigid transform:

$$
f_k(\mathbf{P}) = s \cdot \mathbf{R}(\theta_k)\, \mathbf{P} + \mathbf{t}_k,
$$

where the rotation matrix is

$$
\mathbf{R}(\theta_k) = \begin{pmatrix} \cos\theta_k & -\sin\theta_k \\ \sin\theta_k & \phantom{-}\cos\theta_k \end{pmatrix}.
$$

The scale $s$ is **shared** across all pairs; the rotations $\theta_k$ and translations $\mathbf{t}_k$ are **independent** per pair.

### Objective Function

Minimize the total sum of squared residuals over all pairs and all vertices:

$$
\min_{s,\,\{\theta_k,\,\mathbf{t}_k\}} \; E(s, \{\theta_k, \mathbf{t}_k\}) = \sum_{k=1}^{K} \sum_{i=1}^{N_k} \left\| s\,\mathbf{R}(\theta_k)\,\mathbf{P}_i^{(k)} + \mathbf{t}_k - \mathbf{Q}_i^{(k)} \right\|^2.
$$

This is a system with $N_{\text{tot}}$ point correspondences ($2N_{\text{tot}}$ scalar equations) and 22 unknowns. The problem is overdetermined whenever $2N_{\text{tot}} > 22$, i.e. $N_{\text{tot}} \geq 12$.

---

## Alternating Optimization

The coupling structure is as follows. For **fixed $s$**, each pair $k$ decouples into an independent 2D rigid registration. For **fixed $\{\theta_k, \mathbf{t}_k\}$**, the optimal $s$ has a closed-form expression. This motivates an alternating scheme.

### Preprocessing: Center Each Pair

Before the main loop, compute per-pair centroids and centered coordinates once:

$$
\bar{\mathbf{P}}_k = \frac{1}{N_k}\sum_{i=1}^{N_k} \mathbf{P}_i^{(k)}, \qquad \bar{\mathbf{Q}}_k = \frac{1}{N_k}\sum_{i=1}^{N_k} \mathbf{Q}_i^{(k)},
$$

$$
\tilde{\mathbf{p}}_i^{(k)} = \mathbf{P}_i^{(k)} - \bar{\mathbf{P}}_k, \qquad \tilde{\mathbf{q}}_i^{(k)} = \mathbf{Q}_i^{(k)} - \bar{\mathbf{Q}}_k.
$$

Translation eliminates from the centered objective, so $E$ becomes:

$$
E = \sum_{k=1}^{K} \sum_{i=1}^{N_k} \left\| s\,\mathbf{R}(\theta_k)\,\tilde{\mathbf{p}}_i^{(k)} - \tilde{\mathbf{q}}_i^{(k)} \right\|^2 + \text{(translation terms, solved separately)}.
$$

Also precompute per-pair cross-covariance matrices and source variance:

$$
\mathbf{H}_k = \sum_{i=1}^{N_k} \tilde{\mathbf{p}}_i^{(k)}\,(\tilde{\mathbf{q}}_i^{(k)})^\top \in \mathbb{R}^{2\times 2}, \qquad \sigma_k = \sum_{i=1}^{N_k} \left\|\tilde{\mathbf{p}}_i^{(k)}\right\|^2.
$$

Note that $\mathbf{H}_k$ and $\sigma_k$ depend only on the data and not on the unknowns, so they are computed once and reused in every iteration.

### Algorithm

Initialize $s^{(0)} = 1$.

---

**Loop:** for $m = 0, 1, 2, \ldots$ until convergence:

#### Step A — Update rotations and translations (fix $s$)

For each pair $k = 1, \ldots, K$ independently:

1. Form the scaled cross-covariance:
$$
\mathbf{M}_k = s^{(m)} \cdot \mathbf{H}_k.
$$

2. Compute the SVD: $\mathbf{M}_k = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^\top$.

3. Set the optimal rotation (correcting for reflections):
$$
\mathbf{R}_k^{(m+1)} = \mathbf{V}_k \begin{pmatrix} 1 & 0 \\ 0 & \det(\mathbf{V}_k \mathbf{U}_k^\top) \end{pmatrix} \mathbf{U}_k^\top.
$$

4. Extract angle: $\theta_k^{(m+1)} = \operatorname{atan2}\!\left(R_{k,21}^{(m+1)},\, R_{k,11}^{(m+1)}\right)$.

5. Recover translation:
$$
\mathbf{t}_k^{(m+1)} = \bar{\mathbf{Q}}_k - s^{(m)}\,\mathbf{R}_k^{(m+1)}\,\bar{\mathbf{P}}_k.
$$

#### Step B — Update scale (fix $\{\theta_k, \mathbf{t}_k\}$)

The objective is quadratic in $s$. Setting $\partial E / \partial s = 0$ gives:

$$
s^{(m+1)} = \frac{\displaystyle\sum_{k=1}^{K} \operatorname{tr}\!\left(\mathbf{R}_k^{(m+1)\top} \mathbf{H}_k\right)}{\displaystyle\sum_{k=1}^{K} \sigma_k}.
$$

This is a ratio of total cross-covariance alignment to total source variance, pooled across all pairs. Pairs with more vertices and greater spread contribute more to the scale estimate via both $\mathbf{H}_k$ and $\sigma_k$.

---

### Convergence

Check the relative change in the objective:

$$
\frac{\left| E^{(m+1)} - E^{(m)} \right|}{E^{(m)}} < \varepsilon,
$$

or equivalently monitor $|s^{(m+1)} - s^{(m)}|$. In practice, convergence is reached in **3–5 iterations** since the scale update is smooth and the rotation updates are exact at each step.

### Complexity

Preprocessing costs $O(N_{\text{tot}})$ to compute all $\mathbf{H}_k$ and $\sigma_k$. Each subsequent iteration costs $O(K)$ SVD computations of fixed $2\times 2$ matrices — effectively $O(1)$ each — plus $O(1)$ for the scale update. Total cost per iteration is $O(K)$, dominated by the preprocessing pass.
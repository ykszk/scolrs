# Active Shape Model (ASM) Mathematics Summary

## Standard ASM (Without Transformation)

### Model Definition

**Shape representation:** $\mathbf{x} \in \mathbb{R}^{2n}$

$$\mathbf{x} = [x_1, y_1, x_2, y_2, \ldots, x_n, y_n]^T$$

**Shape model:**

$$\mathbf{x} = \bar{\mathbf{x}} + \mathbf{P}\mathbf{b}$$

Where:
- $\bar{\mathbf{x}} \in \mathbb{R}^{2n}$ : mean shape
- $\mathbf{P} \in \mathbb{R}^{2n \times k}$ : first $k$ eigenvectors (principal components)
- $\mathbf{b} \in \mathbb{R}^k$ : shape parameters
- $n$ : number of landmarks
- $k$ : number of modes

### Individual Landmark Positions

$$x_i = \bar{x}_i + \sum_{j=1}^{k} P_{2i-1,j} b_j$$

$$y_i = \bar{y}_i + \sum_{j=1}^{k} P_{2i,j} b_j$$

### Objective Function (Heatmap Fitting)

**Minimize:**

$$\mathcal{J}(\mathbf{b}) = -\sum_{i=1}^{n} H(x_i, y_i) + \lambda \|\mathbf{b}\|^2$$

Where:
- $H(x_i, y_i)$ : heatmap value at landmark $i$
- $\lambda$ : regularization parameter

### Gradient (General Form)

**Component-wise:**

$$\frac{\partial \mathcal{J}}{\partial b_m} = -\sum_{i=1}^{n} \left( \frac{\partial H}{\partial x_i} \cdot P_{2i-1,m} + \frac{\partial H}{\partial y_i} \cdot P_{2i,m} \right) + 2\lambda b_m$$

**Vector form:**

$$\nabla_{\mathbf{b}} \mathcal{J} = -\mathbf{P}^T \nabla_{\mathbf{x}} H(\mathbf{x}) + 2\lambda \mathbf{b}$$

### Bilinear Interpolation

**Heatmap Value:**

$$H(x_i, y_i) = (1-\alpha)(1-\beta)H_{p,q} + \alpha(1-\beta)H_{p+1,q} + (1-\alpha)\beta H_{p,q+1} + \alpha\beta H_{p+1,q+1}$$

Where:
- $p = \lfloor x_i \rfloor, \quad q = \lfloor y_i \rfloor$ : integer grid coordinates
- $\alpha = x_i - p$ : fractional $x$
- $\beta = y_i - q$ : fractional $y$

**Spatial Gradients:**

$$\frac{\partial H}{\partial x_i} = (1-\beta)(H_{p+1,q} - H_{p,q}) + \beta(H_{p+1,q+1} - H_{p,q+1})$$

$$\frac{\partial H}{\partial y_i} = (1-\alpha)(H_{p,q+1} - H_{p,q}) + \alpha(H_{p+1,q+1} - H_{p+1,q})$$

**Complete Gradient:**

$$\frac{\partial \mathcal{J}}{\partial b_m} = -\sum_{i=1}^{n} \left[ g_{x,i} \cdot P_{2i-1,m} + g_{y,i} \cdot P_{2i,m} \right] + 2\lambda b_m$$

Where:

$$g_{x,i} = \frac{\partial H}{\partial x_i}$$

$$g_{y,i} = \frac{\partial H}{\partial y_i}$$

---

## ASM with Pre-Applied Similarity Transform

### Similarity Transform Definition

$$\mathbf{T}(x, y) = s \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$

Where:
- $s$ : scale
- $\theta$ : rotation angle
- $t_x, t_y$ : translation

### Pre-Transformation (Applied Once Before Optimization)

**Transform Mean Shape:**

$$\bar{\mathbf{x}}' = \mathbf{T}(\bar{\mathbf{x}})$$

For each landmark $i$:

$$\bar{x}'_i = s(\cos\theta \cdot \bar{x}_i - \sin\theta \cdot \bar{y}_i) + t_x$$

$$\bar{y}'_i = s(\sin\theta \cdot \bar{x}_i + \cos\theta \cdot \bar{y}_i) + t_y$$

**Transform Eigenvectors (No Translation!):**

$$\mathbf{P}' = s\mathbf{R}(\theta)\mathbf{P}$$

For each component:

$$P'_{2i-1,m} = s(\cos\theta \cdot P_{2i-1,m} - \sin\theta \cdot P_{2i,m})$$

$$P'_{2i,m} = s(\sin\theta \cdot P_{2i-1,m} + \cos\theta \cdot P_{2i,m})$$

**Note:** Translation is NOT applied to eigenvectors since they represent variations from mean, not absolute positions.

### Transformed ASM Model

After pre-transformation:

$$\mathbf{x} = \bar{\mathbf{x}}' + \mathbf{P}'\mathbf{b}$$

This is mathematically equivalent to:

$$\mathbf{x} = \mathbf{T}(\bar{\mathbf{x}} + \mathbf{P}\mathbf{b})$$

But computationally more efficient!

### Optimization (Same as Standard ASM)

Once transformed, use **identical** optimization:

**Objective:**

$$\mathcal{J}(\mathbf{b}) = -\sum_{i=1}^{n} H(x_i, y_i) + \lambda \|\mathbf{b}\|^2$$

**Gradient:**

$$\frac{\partial \mathcal{J}}{\partial b_m} = -\sum_{i=1}^{n} \left[ g_{x,i} \cdot P'_{2i-1,m} + g_{y,i} \cdot P'_{2i,m} \right] + 2\lambda b_m$$

---

## Key Differences Summary

| Aspect | Standard ASM | Pre-Transformed ASM |
|--------|-------------|---------------------|
| **Mean Shape** | $\bar{\mathbf{x}}$ | $\bar{\mathbf{x}}' = \mathbf{T}(\bar{\mathbf{x}})$ |
| **Eigenvectors** | $\mathbf{P}$ | $\mathbf{P}' = s\mathbf{R}(\theta)\mathbf{P}$ |
| **Model Equation** | $\mathbf{x} = \bar{\mathbf{x}} + \mathbf{P}\mathbf{b}$ | $\mathbf{x} = \bar{\mathbf{x}}' + \mathbf{P}'\mathbf{b}$ |
| **Optimization** | Over $\mathbf{b}$ | Over $\mathbf{b}$ (same!) |
| **Gradient** | Uses $\mathbf{P}$ | Uses $\mathbf{P}'$ |
| **Performance** | N/A | Faster (transform once) |

---

## Equivalence Proof

**Pre-transformed approach:**

$$\mathbf{x} = \bar{\mathbf{x}}' + \mathbf{P}'\mathbf{b}$$

$$\mathbf{x} = \mathbf{T}(\bar{\mathbf{x}}) + s\mathbf{R}(\theta)\mathbf{P}\mathbf{b}$$

**Runtime transformation approach:**

$$\mathbf{x} = \mathbf{T}(\bar{\mathbf{x}} + \mathbf{P}\mathbf{b})$$

$$\mathbf{x} = \mathbf{T}(\bar{\mathbf{x}}) + \mathbf{T}(\mathbf{P}\mathbf{b}) \quad \text{[linearity of } \mathbf{T}\text{]}$$

$$\mathbf{x} = \mathbf{T}(\bar{\mathbf{x}}) + s\mathbf{R}(\theta)\mathbf{P}\mathbf{b} \quad \text{[}\mathbf{T}\text{ doesn't translate variations]}$$

**Therefore:** Pre-transformation $\equiv$ Runtime transformation

But with $O(1)$ computation instead of $O(\text{iterations})$ computation!

---

## Complete Algorithm Summary

### Initialization
1. Load pre-trained ASM: $\bar{\mathbf{x}}, \mathbf{P}$
2. If similarity transform $\mathbf{T}(s, \theta, t_x, t_y)$ is known:
   - Compute $\bar{\mathbf{x}}' = \mathbf{T}(\bar{\mathbf{x}})$
   - Compute $\mathbf{P}' = s\mathbf{R}(\theta)\mathbf{P}$
3. Initialize shape parameters: $\mathbf{b} = \mathbf{0}$

### Optimization Loop (Gradient Descent / Adam)
For each iteration $t = 1, 2, \ldots, T$:

1. **Compute current shape:**
   $$\mathbf{x}^{(t)} = \bar{\mathbf{x}}' + \mathbf{P}'\mathbf{b}^{(t)}$$

2. **For each landmark** $i = 1, \ldots, n$:
   - Extract position: $(x_i, y_i)$
   - Compute bilinear interpolation: $H(x_i, y_i), g_{x,i}, g_{y,i}$

3. **Compute objective:**
   $$\mathcal{J}^{(t)} = -\sum_{i=1}^{n} H(x_i, y_i) + \lambda \|\mathbf{b}^{(t)}\|^2$$

4. **Compute gradient:**
   $$\frac{\partial \mathcal{J}}{\partial b_m} = -\sum_{i=1}^{n} \left[ g_{x,i} \cdot P'_{2i-1,m} + g_{y,i} \cdot P'_{2i,m} \right] + 2\lambda b_m^{(t)}$$

5. **Update parameters:**
   - Using gradient descent: $\mathbf{b}^{(t+1)} = \mathbf{b}^{(t)} - \eta \nabla_{\mathbf{b}} \mathcal{J}$
   - Or using Adam optimizer (adaptive learning rate)

6. **Check convergence:**
   - If $|\mathcal{J}^{(t)} - \mathcal{J}^{(t-1)}| < \epsilon$, stop

### Output
- Optimal shape parameters: $\mathbf{b}^*$
- Fitted landmark positions: $\mathbf{x}^* = \bar{\mathbf{x}}' + \mathbf{P}'\mathbf{b}^*$
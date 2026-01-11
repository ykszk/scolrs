# Bidirectional Iterative Closest Point with Active Shape Model

## Problem Setup

Let:
- **Target points**: $\mathbf{y} = \{\mathbf{y}_j\}_{j=1}^N$ where $\mathbf{y}_j \in \mathbb{R}^d$
- **Active Shape Model (ASM)**: $\mathbf{x}(\mathbf{b}) = \overline{\mathbf{x}} + \mathbf{P}\mathbf{b}$
  - $\overline{\mathbf{x}} \in \mathbb{R}^{Md}$: mean shape (M landmarks, d dimensions)
  - $\mathbf{P} \in \mathbb{R}^{Md \times k}$: matrix of k principal modes
  - $\mathbf{b} \in \mathbb{R}^k$: shape parameters

## Bidirectional ICP-ASM Optimization

### Objective Function

$$\mathbf{b}^* = \arg\min_{\mathbf{b}} E(\mathbf{b})$$

where the bidirectional energy functional is:

$$E(\mathbf{b}) = \alpha \sum_{i=1}^M \|\mathbf{x}_i(\mathbf{b}) - \mathbf{y}_{c_{x \to y}(i)}\|^2 + \beta \sum_{j=1}^N \|\mathbf{x}_{c_{y \to x}(j)}(\mathbf{b}) - \mathbf{y}_j\|^2 + \lambda \|\mathbf{b}\|^2$$

Here:
- **Model-to-Target term** (weight $\alpha$): For each model point, find closest target point
  - $c_{x \to y}(i)$: index of closest target point to model landmark i
- **Target-to-Model term** (weight $\beta$): For each target point, find closest model point
  - $c_{y \to x}(j)$: index of closest model landmark to target point j
- $\lambda > 0$: regularization weight
- $\alpha, \beta \geq 0$: weights for the two correspondence directions

### Iterative Algorithm

**Input**: Target points $\mathbf{y}$, ASM parameters $(\overline{\mathbf{x}}, \mathbf{P})$, weights $(\alpha, \beta, \lambda)$, initial $\mathbf{b}^{(0)}$

**Repeat until convergence**:

1. **Forward Correspondence Step** (Model → Target):
   $$c_{x \to y}^{(t)}(i) = \arg\min_{j=1,\ldots,N} \|\mathbf{x}_i(\mathbf{b}^{(t)}) - \mathbf{y}_j\|^2, \quad i = 1, \ldots, M$$

2. **Backward Correspondence Step** (Target → Model):
   $$c_{y \to x}^{(t)}(j) = \arg\min_{i=1,\ldots,M} \|\mathbf{x}_i(\mathbf{b}^{(t)}) - \mathbf{y}_j\|^2, \quad j = 1, \ldots, N$$

3. **Shape Update Step**: Solve for $\mathbf{b}^{(t+1)}$:
   $$\mathbf{b}^{(t+1)} = \arg\min_{\mathbf{b}} \left[\alpha \sum_{i=1}^M \|\mathbf{x}_i(\mathbf{b}) - \mathbf{y}_{c_{x \to y}^{(t)}(i)}\|^2 + \beta \sum_{j=1}^N \|\mathbf{x}_{c_{y \to x}^{(t)}(j)}(\mathbf{b}) - \mathbf{y}_j\|^2 + \lambda \|\mathbf{b}\|^2\right]$$

### Closed-Form Shape Update

Substituting $\mathbf{x}_i(\mathbf{b}) = \overline{\mathbf{x}}_i + \mathbf{P}_i \mathbf{b}$ where $\mathbf{P}_i$ is the row block of $\mathbf{P}$ corresponding to landmark i:

The gradient with respect to $\mathbf{b}$ is:

$$\frac{\partial E}{\partial \mathbf{b}} = 2\alpha \sum_{i=1}^M \mathbf{P}_i^T (\mathbf{x}_i(\mathbf{b}) - \mathbf{y}_{c_{x \to y}^{(t)}(i)}) + 2\beta \sum_{j=1}^N \mathbf{P}_{c_{y \to x}^{(t)}(j)}^T (\mathbf{x}_{c_{y \to x}^{(t)}(j)}(\mathbf{b}) - \mathbf{y}_j) + 2\lambda \mathbf{b}$$

Setting to zero and solving:

$$\mathbf{b}^{(t+1)} = \mathbf{A}^{-1} \mathbf{r}$$

where:

$$\mathbf{A} = \alpha \mathbf{P}^T \mathbf{P} + \beta \sum_{j=1}^N \mathbf{P}_{c_{y \to x}^{(t)}(j)}^T \mathbf{P}_{c_{y \to x}^{(t)}(j)} + \lambda \mathbf{I}$$

$$\mathbf{r} = \alpha \mathbf{P}^T (\mathbf{y}_c^{x \to y,(t)} - \overline{\mathbf{x}}) + \beta \sum_{j=1}^N \mathbf{P}_{c_{y \to x}^{(t)}(j)}^T (\mathbf{y}_j - \overline{\mathbf{x}}_{c_{y \to x}^{(t)}(j)})$$

Here $\mathbf{y}_c^{x \to y,(t)}$ is the stacked vector of target points corresponding to model landmarks.

### Compact Matrix Form

Define correspondence indicator matrices:
- $\mathbf{W}_{x \to y}^{(t)} \in \{0,1\}^{M \times N}$: $[\mathbf{W}_{x \to y}^{(t)}]_{ij} = 1$ if $c_{x \to y}^{(t)}(i) = j$, else 0
- $\mathbf{W}_{y \to x}^{(t)} \in \{0,1\}^{M \times N}$: $[\mathbf{W}_{y \to x}^{(t)}]_{ij} = 1$ if $c_{y \to x}^{(t)}(k) = i$ for some k mapping to target j

Then:

$$\mathbf{A} = \left(\alpha \mathbf{P}^T \mathbf{P} + \beta \mathbf{P}^T \mathbf{D}_{y \to x}^{(t)} \mathbf{P} + \lambda \mathbf{I}\right)$$

$$\mathbf{r} = \alpha \mathbf{P}^T \mathbf{W}_{x \to y}^{(t)} \tilde{\mathbf{y}} - \alpha \mathbf{P}^T \overline{\mathbf{x}} + \beta \mathbf{P}^T \mathbf{W}_{y \to x}^{(t)T} \tilde{\mathbf{y}} - \beta \mathbf{P}^T \mathbf{D}_{y \to x}^{(t)} \overline{\mathbf{x}}$$

where $\mathbf{D}_{y \to x}^{(t)} \in \mathbb{R}^{M \times M}$ is a diagonal matrix counting how many target points map to each model landmark, and $\tilde{\mathbf{y}}$ is the stacked target points.

### Convergence Criterion

Stop when:
$$\frac{|E(\mathbf{b}^{(t+1)}) - E(\mathbf{b}^{(t)})|}{E(\mathbf{b}^{(t)})} < \epsilon$$

or $\|\mathbf{b}^{(t+1)} - \mathbf{b}^{(t)}\| < \epsilon_b$


# Step-by-Step Derivation of Closed-Form Solution

## Starting Point

We want to minimize:

$$E(\mathbf{b}) = \alpha \sum_{i=1}^M \|\mathbf{x}_i(\mathbf{b}) - \mathbf{y}_{c_{x \to y}(i)}\|^2 + \beta \sum_{j=1}^N \|\mathbf{x}_{c_{y \to x}(j)}(\mathbf{b}) - \mathbf{y}_j\|^2 + \lambda \|\mathbf{b}\|^2$$

## Step 1: Express Model Points in Terms of Shape Parameters

The Active Shape Model gives us:

$$\mathbf{x}(\mathbf{b}) = \overline{\mathbf{x}} + \mathbf{P}\mathbf{b}$$

For individual landmarks (in d dimensions):

$$\mathbf{x}_i(\mathbf{b}) = \overline{\mathbf{x}}_i + \mathbf{P}_i \mathbf{b}$$

where:
- $\overline{\mathbf{x}}_i \in \mathbb{R}^d$: the i-th landmark of the mean shape
- $\mathbf{P}_i \in \mathbb{R}^{d \times k}$: rows $(i-1)d+1$ to $id$ of matrix $\mathbf{P}$

## Step 2: Expand the First Term (Model-to-Target)

$$\alpha \sum_{i=1}^M \|\mathbf{x}_i(\mathbf{b}) - \mathbf{y}_{c_{x \to y}(i)}\|^2$$

Substitute the ASM expression:

$$= \alpha \sum_{i=1}^M \|(\overline{\mathbf{x}}_i + \mathbf{P}_i \mathbf{b}) - \mathbf{y}_{c_{x \to y}(i)}\|^2$$

$$= \alpha \sum_{i=1}^M \|\mathbf{P}_i \mathbf{b} - (\mathbf{y}_{c_{x \to y}(i)} - \overline{\mathbf{x}}_i)\|^2$$

Let $\mathbf{r}_i^{x \to y} = \mathbf{y}_{c_{x \to y}(i)} - \overline{\mathbf{x}}_i$ (residual for landmark i):

$$= \alpha \sum_{i=1}^M \|\mathbf{P}_i \mathbf{b} - \mathbf{r}_i^{x \to y}\|^2$$

Expand the squared norm:

$$= \alpha \sum_{i=1}^M \left[(\mathbf{P}_i \mathbf{b})^T (\mathbf{P}_i \mathbf{b}) - 2(\mathbf{P}_i \mathbf{b})^T \mathbf{r}_i^{x \to y} + (\mathbf{r}_i^{x \to y})^T \mathbf{r}_i^{x \to y}\right]$$

$$= \alpha \sum_{i=1}^M \left[\mathbf{b}^T \mathbf{P}_i^T \mathbf{P}_i \mathbf{b} - 2\mathbf{b}^T \mathbf{P}_i^T \mathbf{r}_i^{x \to y} + \|\mathbf{r}_i^{x \to y}\|^2\right]$$

$$= \alpha \mathbf{b}^T \left(\sum_{i=1}^M \mathbf{P}_i^T \mathbf{P}_i\right) \mathbf{b} - 2\alpha \mathbf{b}^T \left(\sum_{i=1}^M \mathbf{P}_i^T \mathbf{r}_i^{x \to y}\right) + \alpha \sum_{i=1}^M \|\mathbf{r}_i^{x \to y}\|^2$$

Note that $\sum_{i=1}^M \mathbf{P}_i^T \mathbf{P}_i = \mathbf{P}^T \mathbf{P}$:

$$= \alpha \mathbf{b}^T \mathbf{P}^T \mathbf{P} \mathbf{b} - 2\alpha \mathbf{b}^T \mathbf{P}^T \mathbf{r}^{x \to y} + \text{const}$$

where $\mathbf{r}^{x \to y} = [\mathbf{r}_1^{x \to y}; \mathbf{r}_2^{x \to y}; \ldots; \mathbf{r}_M^{x \to y}] \in \mathbb{R}^{Md}$ is the stacked residual vector.

## Step 3: Expand the Second Term (Target-to-Model)

$$\beta \sum_{j=1}^N \|\mathbf{x}_{c_{y \to x}(j)}(\mathbf{b}) - \mathbf{y}_j\|^2$$

For each target point $\mathbf{y}_j$, it corresponds to model landmark $c_{y \to x}(j)$:

$$= \beta \sum_{j=1}^N \|(\overline{\mathbf{x}}_{c_{y \to x}(j)} + \mathbf{P}_{c_{y \to x}(j)} \mathbf{b}) - \mathbf{y}_j\|^2$$

$$= \beta \sum_{j=1}^N \|\mathbf{P}_{c_{y \to x}(j)} \mathbf{b} - (\mathbf{y}_j - \overline{\mathbf{x}}_{c_{y \to x}(j)})\|^2$$

Let $\mathbf{r}_j^{y \to x} = \mathbf{y}_j - \overline{\mathbf{x}}_{c_{y \to x}(j)}$:

$$= \beta \sum_{j=1}^N \|\mathbf{P}_{c_{y \to x}(j)} \mathbf{b} - \mathbf{r}_j^{y \to x}\|^2$$

Expand:

$$= \beta \sum_{j=1}^N \left[\mathbf{b}^T \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{P}_{c_{y \to x}(j)} \mathbf{b} - 2\mathbf{b}^T \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{r}_j^{y \to x} + \|\mathbf{r}_j^{y \to x}\|^2\right]$$

$$= \beta \mathbf{b}^T \left(\sum_{j=1}^N \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{P}_{c_{y \to x}(j)}\right) \mathbf{b} - 2\beta \mathbf{b}^T \left(\sum_{j=1}^N \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{r}_j^{y \to x}\right) + \text{const}$$

## Step 4: Expand the Regularization Term

$$\lambda \|\mathbf{b}\|^2 = \lambda \mathbf{b}^T \mathbf{b} = \lambda \mathbf{b}^T \mathbf{I} \mathbf{b}$$

## Step 5: Combine All Terms

$$E(\mathbf{b}) = \alpha \mathbf{b}^T \mathbf{P}^T \mathbf{P} \mathbf{b} - 2\alpha \mathbf{b}^T \mathbf{P}^T \mathbf{r}^{x \to y}$$
$$+ \beta \mathbf{b}^T \left(\sum_{j=1}^N \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{P}_{c_{y \to x}(j)}\right) \mathbf{b} - 2\beta \mathbf{b}^T \left(\sum_{j=1}^N \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{r}_j^{y \to x}\right)$$
$$+ \lambda \mathbf{b}^T \mathbf{I} \mathbf{b} + \text{const}$$

Group the quadratic and linear terms:

$$E(\mathbf{b}) = \mathbf{b}^T \underbrace{\left[\alpha \mathbf{P}^T \mathbf{P} + \beta \sum_{j=1}^N \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{P}_{c_{y \to x}(j)} + \lambda \mathbf{I}\right]}_{\mathbf{A}} \mathbf{b}$$
$$- 2\mathbf{b}^T \underbrace{\left[\alpha \mathbf{P}^T \mathbf{r}^{x \to y} + \beta \sum_{j=1}^N \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{r}_j^{y \to x}\right]}_{\mathbf{g}} + \text{const}$$

## Step 6: Take the Gradient

The energy is now in the form:

$$E(\mathbf{b}) = \mathbf{b}^T \mathbf{A} \mathbf{b} - 2\mathbf{b}^T \mathbf{g} + \text{const}$$

Taking the gradient with respect to $\mathbf{b}$:

$$\frac{\partial E}{\partial \mathbf{b}} = 2\mathbf{A}\mathbf{b} - 2\mathbf{g}$$

## Step 7: Set Gradient to Zero and Solve

For the minimum:

$$2\mathbf{A}\mathbf{b} - 2\mathbf{g} = \mathbf{0}$$

$$\mathbf{A}\mathbf{b} = \mathbf{g}$$

$$\boxed{\mathbf{b}^* = \mathbf{A}^{-1} \mathbf{g}}$$

## Step 8: Final Closed-Form Solution

### Define the System Matrix

$$\boxed{\mathbf{A} = \alpha \mathbf{P}^T \mathbf{P} + \beta \sum_{j=1}^N \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{P}_{c_{y \to x}(j)} + \lambda \mathbf{I}}$$

### Define the Right-Hand Side Vector

$$\boxed{\mathbf{g} = \alpha \mathbf{P}^T \mathbf{r}^{x \to y} + \beta \sum_{j=1}^N \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{r}_j^{y \to x}}$$

where:
- $\mathbf{r}^{x \to y} = \mathbf{y}_c^{x \to y} - \overline{\mathbf{x}}$ with $\mathbf{y}_c^{x \to y} = [\mathbf{y}_{c_{x \to y}(1)}; \ldots; \mathbf{y}_{c_{x \to y}(M)}]$
- $\mathbf{r}_j^{y \to x} = \mathbf{y}_j - \overline{\mathbf{x}}_{c_{y \to x}(j)}$

### The Solution

$$\boxed{\mathbf{b}^{(t+1)} = \mathbf{A}^{-1} \mathbf{g}}$$

## Step 9: Computational Notes

**For the second term's quadratic part:**
- Each target point $j$ contributes $\mathbf{P}_{c_{y \to x}(j)}^T \mathbf{P}_{c_{y \to x}(j)}$ to $\mathbf{A}$
- If multiple target points map to the same model landmark $i$, the contribution $\mathbf{P}_i^T \mathbf{P}_i$ appears multiple times
- Can be computed efficiently by counting: $\sum_{j=1}^N \mathbf{P}_{c_{y \to x}(j)}^T \mathbf{P}_{c_{y \to x}(j)} = \sum_{i=1}^M n_i \mathbf{P}_i^T \mathbf{P}_i$ where $n_i$ is the number of target points mapping to landmark $i$

**For the second term's linear part:**
- Sum over all target points, each contributing through its corresponding model landmark
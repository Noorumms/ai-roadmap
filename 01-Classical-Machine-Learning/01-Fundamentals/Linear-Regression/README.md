# Linear Regression — Complete Notes

> Univariate → Multivariate → Polynomial, Cost Function, Gradient Descent (fully derived), Learning Rate behavior.

---

## Table of Contents
1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [Univariate Linear Regression](#2-univariate-linear-regression)
3. [The Cost Function — Deep Dive](#3-the-cost-function--deep-dive)
4. [Gradient Descent — Full Derivation From Scratch](#4-gradient-descent--full-derivation-from-scratch)
5. [Why We Subtract (the Negative Sign, Explained Properly)](#5-why-we-subtract-the-negative-sign-explained-properly)
6. [Learning Rate (α) — Big vs Small, In Depth](#6-learning-rate-α--big-vs-small-in-depth)
7. [Multivariate Linear Regression](#7-multivariate-linear-regression)
8. [Feature Scaling](#8-feature-scaling)
9. [Polynomial Regression](#9-polynomial-regression)
10. [The Normal Equation (Alternative to Gradient Descent)](#10-the-normal-equation-alternative-to-gradient-descent)
11. [Worked Numerical Examples](#11-worked-numerical-examples)
12. [Cheat Sheet](#12-cheat-sheet)

---

## 1. What Problem Are We Solving?

Given some input(s) `x` and known output `y`, we want to find a **straight-line (or flat-plane) relationship** between them, so that when a *new*, never-seen `x` comes in, we can predict its `y`.

```
Training Data (x, y pairs)
        |
        v
   Find the best-fitting line/plane   <-- "training"
        |
        v
   Apply that line to a NEW x          <-- "prediction"
        |
        v
      Predicted y
```

**The core assumption of linear regression:** the output changes at a **constant rate** as the input changes — i.e., the relationship really is (approximately) a straight line. If you increase `x` by 1 unit, `y` always changes by the same fixed amount, no matter where you are on the line.

---

## 2. Univariate Linear Regression

"Univariate" = **one** input feature.

### 2.1 The Hypothesis Function

$$h_\theta(x) = \theta_0 + \theta_1 x$$

This is just the school algebra line `y = mx + c`, renamed:

| Algebra | ML | Meaning |
|---|---|---|
| `c` (intercept) | `θ0` | value of `h(x)` when `x = 0` |
| `m` (slope) | `θ1` | how steeply `h(x)` rises per unit of `x` |
| `y` | `hθ(x)` | the model's **predicted** value (not the true `y`) |

- **θ0 and θ1 are called parameters** — the numbers the model is allowed to tune during training.
- If `θ1` increases → line gets steeper.
- If `θ1 = 0` → line is flat, model ignores `x` completely.
- If `θ0` increases → whole line shifts upward, same slope.

### 2.2 Notation You Must Know

| Symbol | Meaning |
|---|---|
| `m` | number of training examples (rows) |
| `x⁽ⁱ⁾, y⁽ⁱ⁾` | the i-th training example (the `(i)` is an **index**, NOT an exponent!) |
| `h(x⁽ⁱ⁾)` | model's prediction for the i-th example |

---

## 3. The Cost Function — Deep Dive

We need a single number that tells us **"how wrong is this particular line, across the whole dataset?"** That number is the **cost function**, `J(θ0, θ1)`.

$$J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$

### 3.1 Building it piece by piece

**Step 1 — Error for one example:**
$$\text{error}^{(i)} = h_\theta(x^{(i)}) - y^{(i)}$$
This is just: *predicted minus actual*.

**Step 2 — Why do we SQUARE the error?**
- If we didn't square it, a model that predicts too high on some points and too low on others could have errors that **cancel out** (e.g., +5 and -5 average to 0), making a genuinely bad model look perfect.
- Squaring makes every error **positive**, so nothing cancels.
- Squaring also **punishes big mistakes much more** than small ones (an error of 10 becomes 100; an error of 2 becomes only 4) — usually exactly what we want.
- Squaring keeps the function **smooth and differentiable everywhere** (unlike absolute value, which has a sharp corner at 0 — bad for calculus-based optimization).

**Step 3 — Sum across ALL training examples:**
$$\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$
Adds up the squared error from every single data point — total "wrongness."

**Step 4 — Why divide by `m`?**
So the cost doesn't automatically get bigger just because you have more data. Dividing by `m` gives you the **average** squared error — a fair, dataset-size-independent measure.

**Step 5 — Why the extra `1/2`?**
Pure calculus convenience. When we differentiate `J` (coming up in Section 4), the power `2` from squaring comes down as a multiplier and **exactly cancels** the `1/2`, leaving a clean formula with no leftover constants. It does **not** change *which* θ values minimize `J` — half of the smallest value is still the smallest value.

### 3.2 What does the cost function look like, geometrically?

For univariate regression (2 parameters, θ0 and θ1), `J(θ0, θ1)` is a **3D bowl shape** — a smooth, convex surface with exactly **one lowest point**.

```
        J(θ0,θ1)
           |        _____
           |      /       \
           |    /           \
           |  /               \
           |/___________________\___ θ1 (and θ0 on the other axis)
                    ^
              global minimum
           (this is what we're hunting for)
```

Because it's convex (one single bowl, no false dips), **any correct optimization method is guaranteed to find the true best θ values** — this is a huge deal, and it's exactly why gradient descent works reliably here.

**Goal, stated precisely:**
$$\min_{\theta_0,\theta_1} J(\theta_0, \theta_1)$$

---

## 4. Gradient Descent — Full Derivation From Scratch

### 4.1 The Big Idea

We can't just "jump" to the bottom of the bowl algebraically for every possible model (too expensive for models with many parameters). Instead, gradient descent takes small, repeated steps **downhill**, using the local slope to decide which direction is downhill.

**Analogy:** standing on a foggy hillside, you can't see the valley floor, but you can *feel* which way the ground slopes under your feet right now. Step slightly in the steepest downhill direction, feel the slope again, step again — eventually you reach the bottom.

### 4.2 The Update Rule

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta_0,\theta_1) \quad \text{(repeat until convergence, for } j = 0, 1\text{)}$$

| Symbol | Meaning |
|---|---|
| `:=` | "gets updated to" (assignment, not equality) |
| `α` (alpha) | learning rate — how big a step to take |
| `∂J/∂θj` | the **partial derivative** (slope) of the cost function w.r.t. that one parameter |
| minus sign | move OPPOSITE to the slope (explained fully in Section 5) |

### 4.3 Deriving the partial derivatives, term by term (the "how do we compute gradient" part)

We need `∂J/∂θ0` and `∂J/∂θ1`. Let's actually do the calculus, not just quote the result.

Start with:
$$J(\theta_0,\theta_1) = \frac{1}{2m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 = \frac{1}{2m}\sum_{i=1}^{m}\left(\theta_0 + \theta_1x^{(i)} - y^{(i)}\right)^2$$

**Derivative w.r.t. θ0:**

Using the chain rule — outer function is `(...)²`, inner function is `(θ0 + θ1x⁽ⁱ⁾ - y⁽ⁱ⁾)`:

$$\frac{\partial}{\partial\theta_0}J = \frac{1}{2m}\sum_{i=1}^{m} 2\left(\theta_0+\theta_1x^{(i)}-y^{(i)}\right)\cdot\frac{\partial}{\partial\theta_0}\left(\theta_0+\theta_1x^{(i)}-y^{(i)}\right)$$

The inner derivative `∂/∂θ0 (θ0 + θ1x⁽ⁱ⁾ - y⁽ⁱ⁾)` = **1** (since θ1x⁽ⁱ⁾ and y⁽ⁱ⁾ don't depend on θ0, and the derivative of θ0 w.r.t. itself is 1).

$$\frac{\partial}{\partial\theta_0}J = \frac{1}{2m}\sum_{i=1}^{m} 2\left(h_\theta(x^{(i)})-y^{(i)}\right)\cdot 1$$

The `2` from the chain rule cancels the `1/2` — **this is exactly why the 1/2 was placed in the cost function in the first place**:

$$\boxed{\frac{\partial}{\partial\theta_0}J = \frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)})-y^{(i)}\right)}$$

**Derivative w.r.t. θ1:**

Same process, but now the inner derivative is `∂/∂θ1 (θ0 + θ1x⁽ⁱ⁾ - y⁽ⁱ⁾) = x⁽ⁱ⁾` (since θ1's coefficient is x⁽ⁱ⁾ itself):

$$\frac{\partial}{\partial\theta_1}J = \frac{1}{2m}\sum_{i=1}^{m} 2\left(h_\theta(x^{(i)})-y^{(i)}\right)\cdot x^{(i)}$$

$$\boxed{\frac{\partial}{\partial\theta_1}J = \frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)})-y^{(i)}\right)\cdot x^{(i)}}$$

**Notice:** θ1's derivative has an *extra* `× x⁽ⁱ⁾` factor compared to θ0's. **Why?** Because θ1 controls the *slope* — its effect on the prediction scales with however large `x` was at that data point. θ0 (the intercept) doesn't depend on `x` at all, so no extra multiplier is needed.

### 4.4 The Final Update Rules

$$\theta_0 := \theta_0 - \alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)})-y^{(i)}\right)$$

$$\theta_1 := \theta_1 - \alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)})-y^{(i)}\right)\cdot x^{(i)}$$

### 4.5 Simultaneous Update — a rule you must not skip

$$\text{temp0} = \theta_0 - \alpha\frac{\partial}{\partial\theta_0}J \qquad \text{temp1} = \theta_1 - \alpha\frac{\partial}{\partial\theta_1}J$$
$$\theta_0 = \text{temp0} \qquad \theta_1 = \text{temp1}$$

**Why compute both into temp variables first, instead of updating θ0 then immediately using the NEW θ0 for θ1's formula?**
Because θ1's gradient is supposed to be computed using the model's state **before** any update this round happened. If you update θ0 first and then plug the *new* θ0 into θ1's calculation, you're mixing old and new model states within a single "step" — mathematically incorrect and can destabilize convergence.

```
FULL GRADIENT DESCENT LOOP

  Guess θ0, θ1 (often start at 0, 0)
        |
        v
  Compute h(x) for every training example
        |
        v
  Compute J(θ0, θ1)   <- how wrong are we overall?
        |
        v
  Compute ∂J/∂θ0 and ∂J/∂θ1   <- the gradients
        |
        v
  Update θ0, θ1 SIMULTANEOUSLY (using temp variables)
        |
        v
  Repeat until J stops decreasing meaningfully (convergence)
```

---

## 5. Why We Subtract (the Negative Sign, Explained Properly)

This is the part people memorize without ever really *feeling* — let's fix that.

The update rule is:
$$\theta_j := \theta_j - \alpha \cdot (\text{slope at current } \theta_j)$$

**Case 1 — slope is positive** (cost function is going *uphill* as θj increases, meaning θj is currently too large):
```
   J(θ)
     |        *  <- you are here, slope points UP-RIGHT (positive)
     |       /
     |      /
     |    ●     <- minimum is to the LEFT of you
     |___________________ θj
```
A positive slope means: increasing θj would make the cost WORSE. So we want to **decrease** θj. Since slope is positive, `θj - α×(positive number)` **decreases** θj. ✅ Correct direction, automatically.

**Case 2 — slope is negative** (cost function is going *downhill* as θj increases, meaning θj is currently too small):
```
   J(θ)
     |  *          <- you are here, slope points DOWN-RIGHT (negative)
     |   \
     |    \
     |     ●       <- minimum is to the RIGHT of you
     |___________________ θj
```
A negative slope means: increasing θj would make the cost BETTER. So we want to **increase** θj. Since slope is negative, `θj - α×(negative number) = θj + α×(positive number)` **increases** θj. ✅ Also correct, automatically.

**The elegance:** the single minus sign handles BOTH situations correctly, without needing an `if` statement to check which direction to move. The sign of the gradient itself always points you the wrong way (uphill); subtracting it always sends you the right way (downhill) — that's the entire trick, and it's why gradient descent is written with a minus sign, always, everywhere you'll ever see it (linear regression, logistic regression, neural networks — all of it).

**One-line summary:** *the gradient points in the direction of steepest INCREASE; we want to DECREASE the cost; so we go the opposite way — hence, minus.*

---

## 6. Learning Rate (α) — Big vs Small, In Depth

α controls **how big a step** you take downhill on every iteration.

### 6.1 α too small

```
  J(θ)
    |  *
    |   *
    |    *
    |     *
    |      *
    |       *  ...(takes forever)... eventually reaches minimum
    |________________________________________ iterations
```
- Each step barely moves θ.
- Gradient descent **works correctly**, but needs a huge number of iterations to actually converge — wastes time/compute.

### 6.2 α too large

```
  J(θ)
    |  *
    |      *              <- jumped clean OVER the minimum
    |            *
    |                  *   <- keeps overshooting, cost gets WORSE
    |________________________________________ iterations
        (J increasing or oscillating = red flag)
```
- Each step **overshoots** the minimum, landing on the opposite side of the bowl — sometimes even higher up than before.
- Symptoms: cost function **increases** between iterations, or bounces up and down (oscillates) instead of smoothly decreasing.
- In the worst case, this **diverges** entirely (cost grows without bound).

### 6.3 α "just right"

```
  J(θ)
    |  *
    |    *
    |      *
    |        *
    |          *___*___*____  (smoothly flattens = converged)
    |________________________________________ iterations
```
Cost decreases quickly AND smoothly every single iteration, flattening out near the minimum.

### 6.4 How to actually pick α in practice

1. **Plot `J` vs. number of iterations** after running gradient descent. This is your primary diagnostic tool.
2. If it's not decreasing every iteration → α is too large → reduce it.
3. If it's decreasing extremely slowly → α is too small → increase it.
4. Try a range of values, increasing roughly 3× each time:
   `0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, ...`
5. Pick the **largest α that still reliably decreases J every iteration** — this gives the fastest *stable* convergence.

**Quick summary table:**

| α value | Symptom | Fix |
|---|---|---|
| Too small | J decreases, but painfully slowly | Increase α |
| Too large | J increases or oscillates | Decrease α |
| Just right | J decreases smoothly, flattens out | Keep it |

---

## 7. Multivariate Linear Regression

"Multivariate" = **more than one** input feature (x1, x2, ..., xn).

### 7.1 Hypothesis (vectorized)

$$h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \dots + \theta_nx_n$$

Introduce a fake feature `x0 = 1` so every term has the same `θⱼxⱼ` shape:

$$h_\theta(x) = \theta_0x_0 + \theta_1x_1+\dots+\theta_nx_n = \theta^Tx$$

This lets the exact same formula scale from 1 feature to thousands, using clean vector notation (`θᵀx` = dot product of the parameter vector and the feature vector).

### 7.2 Cost Function (same shape as before, just with vector θ)

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)})-y^{(i)}\right)^2$$

### 7.3 Gradient Descent — generalizes exactly the same way

$$\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)} \quad \text{simultaneously, for } j=0,1,\dots,n$$

Note: `x0⁽ⁱ⁾ = 1` always, so θ0's update automatically simplifies back to the plain-average form from Section 4.4 — it's the exact same derivation from Section 4.3, just repeated once per feature instead of only twice.

---

## 8. Feature Scaling

**The problem:** if one feature ranges 20–70 (age) and another ranges 30,000–200,000 (income), the cost function's contour plot becomes a **long, thin, elongated ellipse** instead of a nice round bowl — gradient descent then zig-zags slowly instead of heading straight for the minimum.

```
Before scaling:                  After scaling:
  ___________                        _______
 /           \                      /       \
|   long,     |                    |  round, |
|   thin       |                    |  even   |
 \  ellipse   /                      \_______/
  -----------
Gradient descent zig-zags       Gradient descent goes
slowly across it                straight to the center
```

**Two common techniques:**

| Technique | Formula | Range |
|---|---|---|
| Normalization (Min-Max) | `x' = (x-min)/(max-min)` | [0, 1] |
| Standardization (Z-score) | `x' = (x-μ)/σ` | centered at 0, roughly [-3,3] |

Feature scaling is **especially important** once you add polynomial features (Section 9), since `x³` grows far faster than `x`, making the mismatch even worse without scaling.

---

## 9. Polynomial Regression

$$h_\theta(x) = \theta_0+\theta_1x_1+\theta_2x_1^2+\theta_3x_1^3$$

**Why is this still called "linear" regression, when there are squares and cubes in it?**

> It's linear **in the parameters (θ)**, even though it's non-linear in `x`. Relabel `x1²` as a new feature `x2`, and `x1³` as `x3` — the formula becomes exactly `θ0+θ1x1+θ2x2+θ3x3`, which is ordinary multivariate linear regression. All the gradient descent math from Sections 4 and 7 applies completely unchanged — you're just feeding it cleverly engineered features.

**Why do we need this at all?** Not every real relationship is a straight line. A model with only `θ0+θ1x` can never bend to follow a curve. Adding `x²`, `x³` as extra (engineered) features lets the *same* linear regression machinery fit curved relationships.

**Trade-off:** higher-degree polynomials can fit training data extremely well, but risk **overfitting** (an overly wiggly curve chasing noise instead of the real trend) — this is exactly where Regularization becomes necessary (a separate, dedicated topic).

---

## 10. The Normal Equation (Alternative to Gradient Descent)

A **closed-form, one-shot algebraic solution** — no iterations, no learning rate:

$$\theta = (X^TX)^{-1}X^Ty$$

| Symbol | Meaning |
|---|---|
| `X` | matrix of all training examples (rows = examples, columns = features, plus x0=1 column) |
| `y` | vector of all target values |
| `Xᵀ` | X transposed |
| `(XᵀX)⁻¹` | matrix inverse — the "division" step |

**Where does this come from?** Setting every partial derivative of `J(θ)` to **zero simultaneously** (the minimum of a convex bowl is exactly where the slope is zero in every direction) and solving that system of equations algebraically.

**Gradient Descent vs Normal Equation:**

| | Gradient Descent | Normal Equation |
|---|---|---|
| Need to choose α? | Yes | No |
| Iterative? | Yes | No — one-shot |
| Feature scaling needed? | Recommended | Not required |
| Good for large n (features)? | Yes | No — `(XᵀX)⁻¹` costs O(n³), gets very slow |
| Good for large m (examples)? | Yes | Yes |

**Rule of thumb:** use the Normal Equation when n (number of features) is small (roughly under 10,000); use Gradient Descent when n is large.

---

## 11. Worked Numerical Examples

### Example 1 — Computing the Cost

Data: `x = [1,2,3]`, `y = [2,4,6]`. Current θ0=0, θ1=1, so `h(x) = x`.

| x⁽ⁱ⁾ | y⁽ⁱ⁾ | h(x⁽ⁱ⁾) | error |
|---|---|---|---|
| 1 | 2 | 1 | -1 |
| 2 | 4 | 2 | -2 |
| 3 | 6 | 3 | -3 |

$$J = \frac{1}{2(3)}\left[(-1)^2+(-2)^2+(-3)^2\right] = \frac{1}{6}(1+4+9) = \frac{14}{6} \approx 2.33$$

**Common mistake:** forgetting the `1/2m` and reporting the raw sum (14) instead of 2.33.

### Example 2 — One Gradient Descent Step

Same data, α = 0.1, m = 3:

$$\frac{\partial J}{\partial\theta_0} = \frac{1}{3}(-1-2-3) = -2$$
$$\frac{\partial J}{\partial\theta_1} = \frac{1}{3}\left[(-1)(1)+(-2)(2)+(-3)(3)\right] = \frac{-14}{3} \approx -4.67$$

$$\theta_0 := 0 - 0.1(-2) = 0.2$$
$$\theta_1 := 1 - 0.1(-4.67) = 1.467$$

Both θ's **increased** — makes sense, since the model was under-predicting everywhere (all errors negative), so gradient descent correctly pushes the line up and steeper, toward the true relationship `y = 2x`.

### Example 3 — Effect of a Bad Learning Rate

If α were, say, 5 instead of 0.1 in Example 2:
$$\theta_1 := 1 - 5(-4.67) = 1 + 23.35 = 24.35$$
A wildly overshot value — next iteration's error would likely be even bigger than before, illustrating exactly the "α too large" divergence behavior from Section 6.2.

---

## 12. Cheat Sheet

**Hypotheses:**
```
Univariate:      h(x) = θ0 + θ1x
Multivariate:    h(x) = θ0x0 + θ1x1 + ... + θnxn = θᵀx   (x0=1)
Polynomial:      h(x) = θ0 + θ1x + θ2x² + θ3x³ + ...     (still linear in θ)
```

**Cost function:**
```
J(θ) = (1/2m) Σ (h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

**Gradients (derived via chain rule):**
```
∂J/∂θ0 = (1/m) Σ (h(x⁽ⁱ⁾) - y⁽ⁱ⁾)
∂J/∂θj = (1/m) Σ (h(x⁽ⁱ⁾) - y⁽ⁱ⁾) · xⱼ⁽ⁱ⁾      (j ≥ 1)
```

**Update rule (always simultaneous):**
```
θj := θj - α · ∂J/∂θj
```

**Why minus sign:** gradient points toward steepest INCREASE; we want to decrease cost; subtracting moves us the opposite (correct) way — automatically, regardless of whether the slope is positive or negative.

**Learning rate:**
```
Too small → slow convergence (safe but wasteful)
Too large → overshoot / oscillate / diverge (J increases)
Just right → J decreases smoothly every iteration
```

**Feature scaling:** `(x-min)/(max-min)` or `(x-μ)/σ` — speeds convergence, avoids elongated contours, more critical with polynomial features.

**Normal Equation:** `θ = (XᵀX)⁻¹Xᵀy` — no α needed, but O(n³), so only good for small n.

**Convexity guarantee:** the linear regression cost function is always a single smooth bowl (convex) — no false local minima, so gradient descent (with a reasonable α) is always guaranteed to find the true best θ.

# Logistic Regression & Classification Metrics — Complete Notes

> Sigmoid, Hypothesis, Decision Boundary, Cost Function, full Gradient Derivation, Regularization, Multiclass, and every classification metric (Confusion Matrix → Precision/Recall → F1 → ROC-AUC).

---

## Table of Contents
1. [Why Not Just Use Linear Regression?](#1-why-not-just-use-linear-regression)
2. [The Sigmoid Function](#2-the-sigmoid-function)
3. [The Hypothesis Function](#3-the-hypothesis-function)
4. [Decision Boundary](#4-decision-boundary)
5. [Why Squared Error Cost Fails Here](#5-why-squared-error-cost-fails-here)
6. [The Log-Loss Cost Function — Deep Dive](#6-the-log-loss-cost-function--deep-dive)
7. [Gradient Descent — Full Derivation From Scratch](#7-gradient-descent--full-derivation-from-scratch)
8. [Learning Rate — Same Rules as Linear Regression](#8-learning-rate--same-rules-as-linear-regression)
9. [Regularized Logistic Regression](#9-regularized-logistic-regression)
10. [Multiclass Classification (One-vs-All)](#10-multiclass-classification-one-vs-all)
11. [Classification Metrics — Confusion Matrix](#11-classification-metrics--confusion-matrix)
12. [Accuracy, Precision, Recall, F1](#12-accuracy-precision-recall-f1)
13. [ROC Curve & AUC](#13-roc-curve--auc)
14. [Multiclass Metrics](#14-multiclass-metrics)
15. [Worked Numerical Examples](#15-worked-numerical-examples)
16. [Cheat Sheet](#16-cheat-sheet)

---

## 1. Why Not Just Use Linear Regression?

Classification tasks have outputs like Spam/Not-Spam, Pass/Fail — categories, not numbers. You might try: use plain linear regression, and classify anything `≥ 0.5` as class 1.

**Why this fails, concretely:**
- Linear regression's output is **unbounded** — it can predict 5.7 or -3.2, which are meaningless as "probability of being class 1."
- A single extreme outlier can **drag the entire line's slope**, shifting the 0.5 threshold and silently misclassifying points that were correctly classified before.
- The real relationship between features and "is this class 1?" is fundamentally **not a straight line** — it needs to saturate near 0 and 1, not extend infinitely.

We need a function whose output is **always trapped between 0 and 1**, no matter how extreme the input. That function is the **sigmoid**.

---

## 2. The Sigmoid Function

$$g(z) = \frac{1}{1+e^{-z}}$$

| z | g(z) |
|---|---|
| -2 | 0.12 |
| -1 | 0.27 |
| 0 | **0.50** |
| 1 | 0.73 |
| 2 | 0.88 |

```
 g(z)
  1 |              _______________
    |          _.-'
0.5 |......._-'  <- always crosses exactly here at z=0
    |    _.-'
  0 |__.-'________________________
              0                    z
```

- As `z → +∞`, `g(z) → 1` (never quite reaches it).
- As `z → -∞`, `g(z) → 0` (never quite reaches it).
- At `z = 0`, `g(z) = 0.5` exactly — this fact becomes critical for the decision boundary (Section 4).

### 2.1 Derivative of the Sigmoid (needed later for the gradient derivation)

This is worth deriving once, because it's the single piece of calculus that makes everything in Section 7 work out cleanly.

$$g(z) = (1+e^{-z})^{-1}$$

Using the chain rule:
$$g'(z) = -1\cdot(1+e^{-z})^{-2}\cdot(-e^{-z}) = \frac{e^{-z}}{(1+e^{-z})^2}$$

Rewrite `e⁻ᶻ = (1+e⁻ᶻ) - 1`:
$$g'(z) = \frac{(1+e^{-z})-1}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} - \frac{1}{(1+e^{-z})^2} = g(z) - g(z)^2$$

$$\boxed{g'(z) = g(z)\left(1-g(z)\right)}$$

**Why this is beautiful:** the derivative of the sigmoid can be written purely in terms of the sigmoid's own output — no need to recompute `e⁻ᶻ` separately. This clean identity is what makes the gradient descent formula for logistic regression collapse to something surprisingly simple (Section 7).

---

## 3. The Hypothesis Function

$$h_\theta(x) = g(\theta^Tx) = \frac{1}{1+e^{-\theta^Tx}}$$

**What changed vs. linear regression?** We take the exact same linear combination `θᵀx = θ0+θ1x1+...+θnxn` you already know, and pass it **through** the sigmoid.

**What does `hθ(x)` actually mean now?**
$$h_\theta(x) = P(y=1 \mid x;\theta)$$

*"The probability that y=1, given input x and current parameters θ."* This is fundamentally different from linear regression's `h(x)`, which was a direct numeric prediction. Here it's a **probability estimate** — an extra decision step is needed to turn it into an actual class label.

Since there are only 2 classes:
$$P(y=0|x;\theta) = 1 - P(y=1|x;\theta)$$

---

## 4. Decision Boundary

$$h_\theta(x) \geq 0.5 \Rightarrow y=1 \qquad h_\theta(x) < 0.5 \Rightarrow y=0$$

Since `g(z) ≥ 0.5` exactly when `z ≥ 0`, substituting `z = θᵀx`:

$$\theta^Tx \geq 0 \Rightarrow y=1 \qquad \theta^Tx < 0 \Rightarrow y=0$$

**Key idea:** the decision boundary is a property of **θ** (the trained parameters), not of the data itself — once θ is fixed, the boundary is fixed. It's the line/curve where the model "changes its mind."

**Linear boundary example:** θ0=-3, θ1=1, θ2=1 → boundary is `x1+x2=3`, a straight diagonal line.

```
  x2
   3 |  \
     |   \      Predict y=1 here (θᵀx ≥ 0)
   2 |    \
     |     \
   1 |      \
     | Predict y=0 here
   0 |________\____________ x1
     0    1    2    3
```

**Non-linear boundaries:** just like Polynomial Regression, adding engineered features like x1², x2² lets logistic regression produce **curved** boundaries (e.g., a circle: `x1²+x2²=1`). The algorithm is still fundamentally "linear combination + sigmoid" — only the *features* fed into it changed.

---

## 5. Why Squared Error Cost Fails Here

If you plug the sigmoid-based `hθ(x)` into the ordinary squared-error cost from linear regression:

$$J(\theta) = \frac{1}{2m}\sum(h_\theta(x^{(i)})-y^{(i)})^2$$

...something bad happens. Because `hθ(x)` is now a **non-linear** (sigmoid-wrapped) function, this `J(θ)` becomes **non-convex** — a bumpy landscape with multiple local dips, not a single smooth bowl.

```
Linear regression's J:          Logistic regression's J,
  (always convex)                if we (wrongly) use squared error:
      \    /                      \  /\    /\  /
       \  /                        \/  \  /  \/
        \/                          (multiple dips - gradient
   (one clean minimum)               descent can get stuck!)
```

**Why is this dangerous?** Gradient descent could settle into one of the smaller dips, believing it found the best θ, when a much better θ exists elsewhere. We need a **new cost function** that stays convex even with the sigmoid inside it — that's Log Loss.

---

## 6. The Log-Loss Cost Function — Deep Dive

### 6.1 Building it from the two cases

$$\text{Cost}(h_\theta(x),y) = \begin{cases} -\log(h_\theta(x)) & \text{if } y=1 \\ -\log(1-h_\theta(x)) & \text{if } y=0\end{cases}$$

**Intuition for the y=1 case:**
- If `y=1` and the model predicted `hθ(x)` close to **1** (confidently correct) → `-log(1) = 0` → **cost ≈ 0** ✅
- If `y=1` but the model predicted `hθ(x)` close to **0** (confidently WRONG) → `-log(≈0) → ∞` → **cost explodes** ⚠️

```
 Cost when y=1:
    Cost
     |*
     | *
     |   *
     |      *
     |            *___________
     |________________________  hθ(x)
     0          0.5           1
   (max cost near 0)   (min cost near 1)
```

**Real-life analogy:** A weather forecaster who confidently says "0% chance of rain" and then it pours deserves a massive credibility penalty. One who said "45%" gets only a mild penalty for hedging. Log loss encodes exactly "confidently wrong = heavily punished."

### 6.2 The combined (single-formula) version

$$\text{Cost}(h_\theta(x),y) = -y\log(h_\theta(x)) - (1-y)\log(1-h_\theta(x))$$

**Check it:** if y=1, the second term is `(1-1)(...) = 0`, leaving `-log(hθ(x))` — matches. If y=0, the first term vanishes, leaving `-log(1-hθ(x))` — matches. One clean formula covers both cases.

### 6.3 Full cost function (averaged over the dataset)

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right]$$

**Other name for this exact formula: Binary Cross-Entropy** — same math, different community (deep learning literature vs. classical statistics).

---

## 7. Gradient Descent — Full Derivation From Scratch

This is the part most courses skip — let's actually derive it, using the sigmoid derivative from Section 2.1.

We want `∂J/∂θj` for one training example first (then we'll sum/average over all m):

$$\text{Cost} = -y\log(g(z)) - (1-y)\log(1-g(z)) \qquad \text{where } z=\theta^Tx$$

Differentiate w.r.t. θj using the chain rule (`∂z/∂θj = xj`):

$$\frac{\partial \text{Cost}}{\partial\theta_j} = -y\cdot\frac{1}{g(z)}\cdot g'(z)\cdot x_j \;-\; (1-y)\cdot\frac{-1}{1-g(z)}\cdot g'(z)\cdot x_j$$

Substitute `g'(z) = g(z)(1-g(z))` (Section 2.1):

$$= -y\cdot\frac{1}{g(z)}\cdot g(z)(1-g(z))\cdot x_j + (1-y)\cdot\frac{1}{1-g(z)}\cdot g(z)(1-g(z))\cdot x_j$$

The `g(z)` cancels in the first term, the `(1-g(z))` cancels in the second term:

$$= -y(1-g(z))x_j + (1-y)g(z)x_j$$

Expand:
$$= \left[-y + y\cdot g(z) + g(z) - y\cdot g(z)\right]x_j = \left[g(z) - y\right]x_j$$

$$\boxed{\frac{\partial \text{Cost}}{\partial\theta_j} = \left(h_\theta(x) - y\right)x_j}$$

**This is a genuinely remarkable result.** All the messy `log`, sigmoid-derivative, and fraction terms **completely cancel out**, leaving the exact same clean `(prediction - actual) × feature` shape you already derived for linear regression in Section 4.3 of the Linear Regression notes.

Averaging over all `m` examples:

$$\frac{\partial J}{\partial\theta_j} = \frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)}$$

### 7.1 The Final Update Rule

$$\theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)} \quad \text{(simultaneously, for all } j\text{)}$$

**Compare directly to linear regression's update rule** — it's *identical in form*. The only difference is what `hθ(x)` means underneath (sigmoid-wrapped here, plain linear there). This is not a coincidence — it's the direct result of the cancellation you just watched happen above.

**Why the minus sign works here** is identical reasoning to linear regression (see Linear Regression notes, Section 5): the gradient points toward steepest cost increase; we move opposite to it; the sign of the gradient automatically determines whether θj needs to go up or down, so a single subtraction always corrects it in the right direction.

---

## 8. Learning Rate — Same Rules as Linear Regression

Everything from the Linear Regression notes' Section 6 applies unchanged:

| α value | Symptom | Fix |
|---|---|---|
| Too small | J decreases, but painfully slowly | Increase α |
| Too large | J increases or oscillates | Decrease α |
| Just right | J decreases smoothly, flattens out | Keep it |

Plot `J` vs. iterations to diagnose; try values like `0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1`.

**One caveat specific to logistic regression:** because log-loss can shoot toward infinity for confidently-wrong predictions early in training (when θ is still near its random initialization), logistic regression can be *slightly* more sensitive to an overly large α at the very start of training than linear regression is — worth starting a notch smaller if you see erratic behavior in the first few iterations.

---

## 9. Regularized Logistic Regression

Same overfitting problem as linear regression (Lecture 9 concepts) can occur here too — a logistic regression decision boundary can become an overly wiggly, contorted shape (especially with polynomial features) that overfits the training data.

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)}))\right] + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$$

This is simply: **[log-loss cost] + [the same penalty term used in regularized linear regression]**. θ0 is excluded from the penalty, same reasoning as before (it only shifts the boundary's position, doesn't control its "wiggliness").

**Gradient descent update:**

$$\theta_j := \theta_j\left(1-\alpha\frac{\lambda}{m}\right) - \alpha\frac{1}{m}\sum_{i=1}^{m}\left(h_\theta(x^{(i)})-y^{(i)}\right)x_j^{(i)}$$

Identical in form to regularized linear regression — only `hθ(x)`'s meaning differs.

---

## 10. Multiclass Classification (One-vs-All)

Plain logistic regression only answers a single yes/no question — it can't directly handle 3+ classes.

**Fix:** train **N separate binary classifiers**, one per class, each answering "is it THIS class, or everything else?"

$$h_\theta^{(i)}(x) = P(y=i \mid x;\theta) \quad \text{for each class } i$$

**Prediction rule:**
$$\text{prediction} = \max_i \, h_\theta^{(i)}(x)$$

Run all N classifiers on a new point, and pick whichever one reports the highest probability.

```
   New Data Point
        |
   +----+----+----+
Classifier A  Classifier B  Classifier C
   0.85         0.10          0.20
        |
   pick MAX -> Class A
```

---

## 11. Classification Metrics — Confusion Matrix

Once a classifier is trained, we need numbers to judge **how good** it actually is — accuracy alone can be dangerously misleading (see Section 12.4). Everything starts with the **Confusion Matrix**.

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

| Term | Meaning |
|---|---|
| **TP** | Predicted Positive, actually Positive — correct |
| **FP** | Predicted Positive, actually Negative — "false alarm" (Type I error) |
| **TN** | Predicted Negative, actually Negative — correct |
| **FN** | Predicted Negative, actually Positive — "missed it" (Type II error) |

**Memory trick:** the 2nd word (Positive/Negative) is always what the model **predicted**. The 1st word (True/False) tells you whether that prediction was **correct**.

**Concrete example — a medical test for a disease:**
- **FP (Type I error):** test says "sick" but patient is healthy → unnecessary panic/treatment.
- **FN (Type II error):** test says "healthy" but patient is actually sick → a **dangerous** missed diagnosis.
Different applications care about FP and FN very differently — this motivates why Accuracy alone isn't enough (see below).

---

## 12. Accuracy, Precision, Recall, F1

### 12.1 Accuracy

$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}$$

*"Out of everything, what fraction did we get right?"* Simple, but dangerously misleading on **imbalanced datasets** (see 12.4).

### 12.2 Precision

$$\text{Precision} = \frac{TP}{TP+FP}$$

*"Out of everything we PREDICTED as positive, how many actually were?"* High precision = few false alarms.

**When to prioritize Precision:** spam detection — you'd rather let a spam email slip into the inbox (FN) than wrongly send an important email to spam (FP).

### 12.3 Recall (a.k.a. Sensitivity, True Positive Rate)

$$\text{Recall} = \frac{TP}{TP+FN}$$

*"Out of everything that WAS actually positive, how many did we catch?"* High recall = few missed cases.

**When to prioritize Recall:** cancer/disease screening — you'd rather flag a few healthy people for extra tests (FP) than miss an actual sick patient (FN).

### 12.4 Why Accuracy Alone Can Lie to You

Imagine a dataset with 950 healthy patients and 50 sick patients (imbalanced). A lazy model that predicts "healthy" for **everyone** achieves:

$$\text{Accuracy} = \frac{950}{1000} = 95\%$$

...despite **catching zero actual sick patients** (Recall = 0%). This is exactly why Precision and Recall exist — they expose failure modes that a single "% correct" number hides completely.

### 12.5 The Precision-Recall Trade-off

Precision and Recall usually pull in **opposite directions** — raising the classification threshold (e.g., requiring `hθ(x) ≥ 0.9` instead of `≥ 0.5` to predict positive) makes the model more "cautious":
- Fewer false alarms → **Precision goes up**
- But more real positives get missed → **Recall goes down**

```
 Precision
    |  \
    |   \
    |    \___
    |         \___
    |______________\___ Recall
   (as threshold increases: Precision↑, Recall↓ — a trade-off curve)
```

### 12.6 F1 Score — the balance between the two

$$F1 = 2 \times \frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}$$

This is the **harmonic mean** of Precision and Recall (not the simple average) — chosen specifically because the harmonic mean punishes a big imbalance between the two much more than a plain average would. If either Precision or Recall is very low, F1 will also be low, even if the other is perfect.

**Use F1 when:** you need one single number balancing both concerns, especially on imbalanced datasets where accuracy alone is misleading.

### 12.7 Specificity (bonus, completes the picture)

$$\text{Specificity} = \frac{TN}{TN+FP}$$

*"Out of everything that WAS actually negative, how many did we correctly call negative?"* The negative-class mirror of Recall.

---

## 13. ROC Curve & AUC

**Key term — Threshold:** the cutoff probability (default 0.5) above which we predict class 1. Changing this threshold changes every metric above.

**ROC Curve (Receiver Operating Characteristic):** plots **True Positive Rate (Recall)** against **False Positive Rate** at every possible threshold value.

$$\text{False Positive Rate (FPR)} = \frac{FP}{FP+TN}$$

```
 TPR (Recall)
   1 |        ________________
     |      /
     |    /              <- ideal: hugs the top-left corner
     |  /       (a random-guess classifier
     |/          would just follow the diagonal)
   0 |________________________ FPR
     0                        1
```

**Key term — AUC (Area Under the Curve):** a single number summarizing the ROC curve — the probability that the model ranks a randomly chosen positive example higher than a randomly chosen negative one.

| AUC value | Meaning |
|---|---|
| 1.0 | Perfect classifier |
| 0.5 | No better than random guessing |
| < 0.5 | Worse than random (something is likely wrong) |

**Why use ROC-AUC over accuracy?** AUC evaluates the model **across all possible thresholds at once**, rather than being locked to a single cutoff (0.5) — useful when you haven't yet decided the "right" threshold for your specific application, or want a threshold-independent comparison between two models.

---

## 14. Multiclass Metrics

Precision/Recall/F1 were defined for binary classification — for multiclass (Section 10), each class gets its own Precision/Recall/F1 (treating "this class" as positive, "all others" as negative — literally the One-vs-All framing again). These are then combined:

| Averaging method | How it works | When to use |
|---|---|---|
| **Macro average** | Simple average of each class's metric, treating all classes equally | When all classes matter equally, regardless of size |
| **Weighted average** | Average weighted by how many examples each class has | When class imbalance should be reflected in the summary |
| **Micro average** | Pool all TP/FP/FN across classes first, then compute one global metric | When you care about overall performance across all individual predictions |

**Multiclass Confusion Matrix:** becomes an N×N grid instead of 2×2 — diagonal = correct predictions, off-diagonal = specific types of confusion between class pairs (useful for spotting which classes the model mixes up most).

---

## 15. Worked Numerical Examples

### Example 1 — Sigmoid & Prediction

θ0=-3, θ1=1, θ2=1. New point: x1=4, x2=1.

$$z = -3+(1)(4)+(1)(1) = 2 \qquad h_\theta(x) = g(2) = \frac{1}{1+e^{-2}} \approx 0.88$$

Since `0.88 ≥ 0.5` → **predict y=1**, 88% confidence.

### Example 2 — Confusion Matrix → Metrics

A test set of 100 examples gives: TP=40, FP=10, FN=5, TN=45.

$$\text{Accuracy} = \frac{40+45}{100} = 0.85 = 85\%$$
$$\text{Precision} = \frac{40}{40+10} = 0.80 = 80\%$$
$$\text{Recall} = \frac{40}{40+5} = 0.889 = 88.9\%$$
$$F1 = 2\times\frac{0.80\times0.889}{0.80+0.889} = 2\times\frac{0.711}{1.689} \approx 0.842 = 84.2\%$$

**Interpretation:** the model catches most real positives well (high recall) but has a slightly higher false-alarm rate (precision a bit lower) — worth knowing which of the two matters more for the actual application before declaring this "good."

### Example 3 — One Gradient Descent Step (Logistic Regression)

θ1 = 0.5 (before update), gradient term (already averaged across m) = 0.3, α = 0.1.

$$\theta_1 := 0.5 - 0.1(0.3) = 0.5 - 0.03 = 0.47$$

Same mechanical update as linear regression — only the `hθ(x)` used inside that gradient term was sigmoid-based.

---

## 16. Cheat Sheet

**Core formulas:**
```
Sigmoid:        g(z) = 1/(1+e⁻ᶻ)
Sigmoid deriv:  g'(z) = g(z)(1-g(z))
Hypothesis:     hθ(x) = g(θᵀx) = P(y=1|x;θ)
Decision rule:  hθ(x) ≥ 0.5  ⟺  θᵀx ≥ 0  →  y=1
Cost (1 ex.):   -y·log(hθ(x)) - (1-y)·log(1-hθ(x))
Full cost J(θ): -(1/m) Σ [y·log(hθ(x)) + (1-y)·log(1-hθ(x))]
Gradient:       ∂J/∂θj = (1/m) Σ (hθ(x⁽ⁱ⁾)-y⁽ⁱ⁾)·xⱼ⁽ⁱ⁾   (same shape as linear regression!)
Update:         θj := θj - α · ∂J/∂θj      (simultaneous)
Regularized:    θj := θj(1-αλ/m) - α(1/m)Σ(hθ(x)-y)xⱼ
```

**Why log-loss instead of squared error:** squared error + sigmoid → non-convex J → risk of local minima. Log-loss stays convex.

**Why the gradient formula looks identical to linear regression:** the sigmoid derivative `g(z)(1-g(z))` cancels perfectly against the `1/g(z)` and `1/(1-g(z))` terms from differentiating the logs — see Section 7's full derivation.

**Classification metrics summary:**
```
Confusion Matrix:  TP, FP, FN, TN  (2nd word = predicted, 1st word = correct or not)

Accuracy  = (TP+TN)/(TP+TN+FP+FN)     -- misleading on imbalanced data
Precision = TP/(TP+FP)                 -- "of predicted positives, how many right?" -- minimize false alarms
Recall    = TP/(TP+FN)                 -- "of actual positives, how many caught?"   -- minimize missed cases
F1        = 2·(P·R)/(P+R)              -- harmonic mean, balances precision & recall
Specificity = TN/(TN+FP)               -- recall's mirror, for the negative class
AUC       = area under ROC curve       -- 0.5=random, 1.0=perfect, threshold-independent
```

**When to prioritize which metric:**
| Situation | Prioritize |
|---|---|
| Spam filter (false alarm = bad) | Precision |
| Disease screening (missed case = bad) | Recall |
| Balanced importance, need one number | F1 |
| Comparing models across all thresholds | AUC |
| Balanced classes, simple summary | Accuracy (only then it's safe) |

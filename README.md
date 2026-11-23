# Psyber Mentalytics: Mathematical Modeling of Mental Health Risk Over Time

These notebooks model how social media use moves people between mental health risk levels over time. They also find the education and intervention plan that reduces that risk most for the lowest cost. This is the code behind my ICMSC 2026 presentation, *Psyber Mentalytics: A Novel Deep Learning Algorithm to Predict Mental Health Risks Due to Social Media, and Mathematical Modeling of Variability Over Time*.

**Author:** Achyutan Veeraraghavan
**Mentor:** Dr. Padmanabhan Seshaiyer (George Mason University)

## The model

The NLMH model is an SEIR-style compartmental model with four risk levels. Each one is a fraction of the population, so they always add up to 1.

| Compartment | Meaning |
|---|---|
| N | No risk (protected through education) |
| L | Low risk (susceptible) |
| M | Medium risk |
| H | High risk |

People can only move one level at a time. Risk goes up through β₁ (L → M) and β₂ (M → H), and protection fades through δ (N → L). Recovery goes the other way through α (L → N), γ₁ (M → L) and γ₂ (H → M).

There are two versions of the ODEs:

- **Linear:** people move at a fixed rate, for example β₁L.
- **Non-linear:** movement depends on contact with riskier people (β₁L(M + H) and β₂MH). This is the social contagion version.

The control is a single number αw(t) between 0 and 1 for how much education and intervention is applied each week. It scales all three recovery rates: α = vα·αw, γ₁ = v₁·αw, γ₂ = v₂·αw.

## Cost functions

Each ODE version is solved with two cost functions, which gives four combinations.

Quadratic cost (the original):

$$V(\alpha_w) = \int_0^T \left(\omega_1 M(t) + \omega_2 H(t) + \frac{\lambda_w}{2}\alpha_w(t)^2\right)dt$$

Linear cost:

$$V(\alpha_w) = \int_0^T \left(\omega_1 M(t) + \omega_2 H(t) + \lambda_w\,\alpha_w(t)\right)dt$$

The quadratic cost gives a smooth control with a closed-form formula. The linear cost gives a bang-bang control with singular arcs. On a singular arc the switching function Φ stays at 0, and the control is found from its second derivative: αw = −A₂/B₂.

## Notebooks

### Stability analysis

| Notebook | What it does |
|---|---|
| `Eigenvalue_Stability_Analysis_Linear.ipynb` | Routh-Hurwitz test on the linear model's constant Jacobian. Stable for all αw > 0. |
| `Eigenvalue_Stability_Analysis_NonLinear_DFE.ipynb` | Routh-Hurwitz test at the disease-free equilibrium. Stable once αw ≥ 0.1235. |
| `Eigenvalue_Stability_Analysis_NonLinear_END.ipynb` | Stability of the endemic equilibrium, which only exists for αw < 0.1142. Neutrally stable there: A₃ = 0 because β₂M* = γ₂. |

### Optimal control (Pontryagin's Minimum Principle + forward-backward sweep)

| Notebook | Cost | ODEs |
|---|---|---|
| `Quadratic_Optimal_Control_Forward_Backward_Sweeping_Linear_ODE.ipynb` | Quadratic | Linear |
| `Quadratic_Optimal_Control_Forward_Backward_Sweeping_NonLinear_ODE.ipynb` | Quadratic | Non-linear |
| `Linear_Optimal_Control_Forward_Backward_Sweeping_Linear_ODE.ipynb` | Linear | Linear |
| `Linear_Optimal_Control_Forward_Backward_Sweeping_NonLinear_ODE.ipynb` | Linear | Non-linear |

Each control notebook follows the same steps: Hamiltonian, optimality condition, adjoint equations, transversality conditions (Wᵢ(T) = 0), and the forward-backward sweep.

The quadratic notebooks update αw pointwise with a 0.5 blend each sweep. The linear notebooks use the same forward and backward sweeps, but a pointwise bang-bang update never settles on a singular arc. So they use Φ as the gradient in bounded L-BFGS-B over αw ∈ [0, 1], and then check Φ = 0 and αw = −A₂/B₂ on the arc.

## Results

Initial conditions come from my earlier MLP classifier: N(0) = 0, L(0) = M(0) = 0.268, H(0) = 0.464, over T = 52 weeks.

| | Quadratic / Linear | Quadratic / Non-Linear | Linear / Linear | Linear / Non-Linear |
|---|---|---|---|---|
| Baseline V (no control) | 229.8590 | 219.3598 | 229.8590 | 219.3598 |
| Optimal V* | 39.5207 | 21.7449 | 59.5439 | 32.7362 |
| Burden reduction | 82.8% | 90.1% | 74.1% | 85.1% |
| Iterations | 14 | 15 | 761 | 422 |
| Tolerance | 10⁻⁴ | 10⁻³ | 10⁻¹⁵ | 10⁻¹⁵ |

Singular arcs with linear cost:

- **Linear ODEs:** weeks 19.0–40.1, with αw going from about 0.60 to 0.51
- **Non-linear ODEs:** weeks 15.5–23.2, with αw going from about 0.17 to 0.14

The non-linear model reduces burden more because treating high-risk people also slows the spread to everyone else.

## Parameters

Rates are per week.

| Parameter | Meaning | Value |
|---|---|---|
| δ | Protection decay rate (N → L) | 0.02 |
| β₁, β₂ | Escalation rates (L → M, M → H) | 0.15, 0.10 |
| vα, v₁, v₂ | Weights splitting αw into α, γ₁, γ₂ | 0.40, 0.35, 0.25 |
| ω₁, ω₂ | Societal cost of medium and high risk | 1.0, 5.0 |
| λw | Financial cost weight of the intervention | 1.0 |
| αw | Composite control | [0, 1] |

## Running the notebooks

They run in Google Colab or any Jupyter setup with Python 3.

```
pip install numpy scipy matplotlib
```

Each notebook runs top to bottom and saves its figures as PNGs. The control notebooks name them by case, for example `quad_cost_lin_ode_states.png` or `lin_cost_nonlin_ode_switching.png`.

## Related work

- **[Mind the Gap](https://github.com/achyutanveera/Social-Media-Wellness-Check-In-App):** a student wellness check-in app built on the same Psyber Mentalytics classifier. The plan is to use months of check-in data from it to estimate β and γ with physics-informed neural networks (PINNs) instead of relying on literature estimates.

## References

1. Ogueda-Oliva, Alonso, and Padmanabhan Seshaiyer. *Quantitative Biology: Mathematical Modeling and Computation*. Academic Press, 2026.
2. Joshi, Hem Raj, et al. "Optimal control of an SIR model with changing behavior through an education campaign." (2015).
3. Fleming, Wendell H., and Raymond W. Rishel. *Deterministic and Stochastic Optimal Control*. Springer-Verlag, 1975.
4. Krener, Arthur J. "The High Order Maximal Principle and Its Application to Singular Extremals." *SIAM Journal on Control and Optimization* 15.2 (1977): 256-293.

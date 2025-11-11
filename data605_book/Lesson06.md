#### Logic-Based AI Under Uncertainty: Problem

- Binomial Distribution

::::{grid} 12
:gutter: 2

:::{grid-item}
:columns: 5

Probability of $k$ heads out of $n$ tosses given $p$ is:
:::

:::{grid-item}
:columns: 7

```{math}
:nowrap:
\begin{align*}
  & X \sim \mathrm{Binomial}(n, p) \\
  & \Pr(k) = \frac{n!}{k! (n - k)!}\, p^k (1 - p)^{n-k}
\end{align*}
```

![](lectures_source/figures/Lesson07_Binomial_distribution.png)
:::
::::

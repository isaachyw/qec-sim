Implement a Monte Carlo estimator for quantum circuits using stabilizer channel decompositions.

Inputs:
- rho decomposition:
    rho = sum_i q0[i] * sigma[i]

- For each gate k = 1..K:
    chi[k] = sum_i qk[k][i] * S[k][i]
  where S[k][i] are stabilizer channels.

- Observable decomposition:
    phi = sum_i q_obs[i] * sigma[i]

- Number of Monte Carlo samples N.

Algorithm:

1. For each decomposition level k = 0..K+1:
    Compute probabilities:
        p[k][i] = |q[k][i]| / sum_j |q[k][j]|

2. Initialize estimate F = 0.

3. Repeat N times:
    a. Sample index i_k from p[k] for all k.

    b. Set rho_star = sigma[i_0].

    c. For k = 1..K:
           rho_star = apply_channel(S[k][i_k], rho_star)

    d. Compute weight:
           w = product_k q[k][i_k] / product_k p[k][i_k]

    e. Compute observable value:
           f = trace( sigma[i_{K+1}] * rho_star )

    f. Accumulate:
           F += w * f / N

4. Return F.

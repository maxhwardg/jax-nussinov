"""
Basic implementation of Nussinov algorithm in JAX
Uses a fixed sequence unlike the probabilistic version.
Derivatives can be taken w.r.t. the weight matrices.
"""
import jax
import jax.numpy as jnp
import numpy as np


def make_jax_nussinov(n: int, min_hairpin: int = 0):
    """Make a JAX Nussinov function for a sequence of length n"""
    # TODO: Add checkpointing. Currently this uses for O(n^3) memory for gradient calculation
    @jax.jit
    def nussinov(bp_weights, unpaired_weights, per_nt_scale=1.0):
        seq_inds = jnp.arange(n)

        def process_i(i_: int, dp_table):
            """Process i-th row of dp_table""" 
            # Iterate backwards through i
            i = n - i_ - 1

            def process_j(j: int):
                """Process j-th element of i-th row"""
                unpaired_sm = jax.lax.select(i+1 > j, 1.0, dp_table[i+1, j])\
                    * unpaired_weights[i] * per_nt_scale
                def process_k(k: int):
                    ip1 = jax.lax.select(
                        i+1 > k-1, 1.0, dp_table[i+1, k-1])
                    kp1 = jax.lax.select(k+1 > j, 1.0, dp_table[k+1, j])
                    res = bp_weights[i, k] * ip1 * kp1
                    res = jax.lax.select(jnp.logical_and(
                        i + min_hairpin < k, k <= j), res, 0.0)
                    return res * per_nt_scale * per_nt_scale
                paired_sm = jnp.sum(jax.vmap(process_k)(seq_inds))
                res = jax.lax.select(j >= i, unpaired_sm+paired_sm, 0.0)
                return res

            return dp_table.at[i].set(jax.vmap(process_j)(seq_inds))
        init_dp = jnp.zeros((n, n))
        return jax.lax.fori_loop(0, n, process_i, init_dp)[0, n-1]
    return nussinov


def standard_nussinov_partition_fn(bp_weights, unpaired_weights, min_hairpin: int = 0):
    """Classical implementation of Nussinov algorithm"""
    n = len(unpaired_weights)
    assert n == len(bp_weights)
    dp = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n-1, -1, -1):
        for j in range(i, n):
            dp[i][j] = unpaired_weights[i] * (1.0 if i+1 > j else dp[i+1][j])
            for k in range(i+min_hairpin+1, j+1):
                kp1 = 1.0 if k+1 > j else dp[k+1][j]
                ip1 = 1.0 if i+1 > k-1 else dp[i+1][k-1]
                dp[i][j] += bp_weights[i][k] * kp1 * ip1
    return float(dp[0][n-1])

def standard_nussinov_max(bp_weights, unpaired_weights, min_hairpin: int = 0):
    """Classical implementation of Nussinov algorithm"""
    n = len(unpaired_weights)
    assert n == len(bp_weights)
    dp = [[-1e9 for _ in range(n)] for _ in range(n)]
    for i in range(n-1, -1, -1):
        for j in range(i, n):
            dp[i][j] = unpaired_weights[i] + (0.0 if i+1 > j else dp[i+1][j])
            for k in range(i+min_hairpin+1, j+1):
                kp1 = 0.0 if k+1 > j else dp[k+1][j]
                ip1 = 0.0 if i+1 > k-1 else dp[i+1][k-1]
                dp[i][j] = max(dp[i][j], bp_weights[i][k] + kp1 + ip1)
    return float(dp[0][n-1])


def random_weights(n: int):
    bp_weights = np.random.normal(size=(n, n))
    bp_weights = np.triu(bp_weights, k=1)
    unpaired_weights = np.random.normal(size=n)
    return bp_weights, unpaired_weights


def fuzz_test(min_hairpin: int = 0):
    for n in range(1, 16):
        nuss = make_jax_nussinov(n, min_hairpin)
        for epoch in range(3):
            bp_weights, unpaired_weights = random_weights(n)
            jax_res = nuss(bp_weights, unpaired_weights)
            std_res = standard_nussinov_partition_fn(
                bp_weights, unpaired_weights, min_hairpin)
            assert np.allclose(
                jax_res, std_res, atol=1e-5), f"n={n} epoch={epoch} jax={jax_res} std={std_res}"
            print(f"Epoch passed: n={n} epoch={epoch}")
    print("All tests passed!")



def main():
    """
    Runs tests and demos for the fixed sequence Nussinov implementation
    """
    # Run fuzz tests to check correctness
    print("Running fuzz tests...")
    fuzz_test(min_hairpin=0)
    fuzz_test(min_hairpin=3)

    # Compute gradients w.r.t. logits
    print("Computing gradients w.r.t. logits...")
    n = 10
    nuss = make_jax_nussinov(n)
    bp_weights, unpaired_weights = random_weights(n)

    # Compute gradients w.r.t. bp_weights and unpaired_weights
    g_weights = jax.grad(nuss, argnums=[0, 1])(bp_weights, unpaired_weights)
    print("Grads wrt weights:", g_weights)
    
    # Example of per-nucleotide scaling
    # Since the partition function can be very large, we use a per-nucleotide scaling factor f
    # This computes the partition function but scaled by f**n
    # So, we compute Z*f^n instead of Z
    # This is useful to prevent numerical instability
    print("Per-nucleotide scaling example:")
    n = 400
    # Using exp to scale these as though they are free energies
    bp_weights = np.exp(np.random.normal(size=(n, n)))
    unpaired_weights = np.exp(np.random.normal(size=n))
    nuss = make_jax_nussinov(n)
    grad_fn = jax.grad(nuss, argnums=[0, 1])
    # This needs to be chosen carefully to avoid underflow/overflow
    # This was chosen empirically
    scale = 0.27
    jax_res = nuss(bp_weights, unpaired_weights, per_nt_scale=scale)
    g_bp_weights, g_unpaired_weights = grad_fn(bp_weights, unpaired_weights, per_nt_scale=scale)
    print("JAX Nussinov result:", jax_res, "Grads: ", g_unpaired_weights)
    
    


if __name__ == "__main__":
    main()

"""Basic implementation of Nussinov algorithm in JAX"""
import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np


def make_jax_nussinov(n: int, min_hairpin: int = 0):
    """Make a JAX Nussinov function for a sequence of length n"""
    # TODO: Add checkpointing. Currently this uses for O(n^3) memory for gradient calculation
    # TODO: Add per-nucleotide scaling to prevent numerical instability
    @jax.jit
    def nussinov(base_logits, bp_weights, unpaired_weights):
        base_probs = jnn.softmax(base_logits, axis=1)
        seq_inds = jnp.arange(n)
        b_ids = jnp.arange(4)
        bp_ids = jnp.arange(4*4)

        def process_i(i_: int, dp_table):
            """Process i-th row of dp_table"""
            # Iterate backwards through i
            i = n - i_ - 1

            def process_j(j: int):
                """Process j-th element of i-th row"""

                def process_unpaired(b_id):
                    """Unpaired nt at i"""
                    dp_val = jax.lax.select(i+1 > j, 1.0, dp_table[i+1, j])
                    return base_probs[i, b_id] * unpaired_weights[b_id] * dp_val

                def process_k(k: int):
                    def process_pair(bp_id):
                        """Pair (i, k)"""
                        b1 = bp_id // 4
                        b2 = bp_id % 4
                        ip1 = jax.lax.select(
                            i+1 > k-1, 1.0, dp_table[i+1, k-1])
                        kp1 = jax.lax.select(k+1 > j, 1.0, dp_table[k+1, j])
                        return base_probs[i, b1] * base_probs[j, b2] * bp_weights[b1, b2] * ip1 * kp1
                    res = jnp.sum(jax.vmap(process_pair)(bp_ids))
                    res = jax.lax.select(jnp.logical_and(
                        i + min_hairpin < k, k <= j), res, 0.0)
                    return res
                unpaired_sm = jnp.sum(jax.vmap(process_unpaired)(b_ids))
                paired_sm = jnp.sum(jax.vmap(process_k)(seq_inds))
                res = jax.lax.select(j >= i, unpaired_sm+paired_sm, 0.0)
                return res

            return dp_table.at[i].set(jax.vmap(process_j)(seq_inds))
        init_dp = jnp.zeros((n, n))
        return jax.lax.fori_loop(0, n, process_i, init_dp)[0, n-1]
    return nussinov


def standard_nussinov(base_probs, bp_weights, unpaired_weights, min_hairpin: int = 0):
    """Classical implementation of Nussinov algorithm"""
    n = len(base_probs)
    dp = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n-1, -1, -1):
        for j in range(i, n):
            dp[i][j] = 0.0
            for b in range(4):
                dp[i][j] += base_probs[i][b] * unpaired_weights[b] * \
                    (1.0 if i+1 > j else dp[i+1][j])
            for k in range(i+min_hairpin+1, j+1):
                for b1 in range(4):
                    for b2 in range(4):
                        kp1 = 1.0 if k+1 > j else dp[k+1][j]
                        ip1 = 1.0 if i+1 > k-1 else dp[i+1][k-1]
                        dp[i][j] += base_probs[i][b1] * \
                            base_probs[j][b2] * bp_weights[b1][b2] * kp1 * ip1
    return float(dp[0][n-1])


def fuzz_test(min_hairpin: int = 0):
    for n in range(1, 16):
        nuss = make_jax_nussinov(n, min_hairpin)
        for epoch in range(3):
            logits = np.random.normal(size=(n, 4))
            probs = np.exp(logits) / np.sum(np.exp(logits),
                                            axis=1, keepdims=True)
            bp_weights = np.random.normal(size=(4, 4))
            unpaired_weights = np.random.normal(size=(4))
            jax_res = nuss(logits, bp_weights, unpaired_weights)
            std_res = standard_nussinov(probs, bp_weights, unpaired_weights, min_hairpin)
            assert np.allclose(
                jax_res, std_res, atol=1e-5), f"n={n} epoch={epoch} jax={jax_res} std={std_res}"
            print(f"Epoch passed: n={n} epoch={epoch}")
    print("All tests passed!")


def main():

    # Run fuzz tests to check correctness
    fuzz_test(min_hairpin=0)
    fuzz_test(min_hairpin=3)

    # Compute gradients w.r.t. logits
    n = 10
    nuss = make_jax_nussinov(n)
    logits = np.random.normal(size=(n, 4))
    bp_weights = np.random.normal(size=(4, 4))
    unpaired_weights = np.random.normal(size=(4))
    g_logits = jax.grad(nuss, argnums=0)(logits, bp_weights, unpaired_weights)
    print("Grads wrt logits:", g_logits)

    # Compute gradients w.r.t. bp_weights and unpaired_weights
    g_weights = jax.grad(nuss, argnums=[1, 2])(
        logits, bp_weights, unpaired_weights)
    print("Grads wrt weights:", g_weights)


if __name__ == "__main__":
    main()

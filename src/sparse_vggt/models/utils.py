def print_sparse_info(
    sparse_ratio,
    cdf_threshold,
    pool_mode,
    use_hilbert,
    aux_output,
    aux_sparsity_only,
):
    # Print some info
    print("-" * 100)

    if sparse_ratio is not None:
        print(f"sparse_ratio: {sparse_ratio}")
    if cdf_threshold is not None:
        print(f"cdf_threshold: {cdf_threshold}")

    if use_hilbert:
        print("Using Hilbert permutation")
    else:
        print("Not using Hilbert permutation")

    print(f"Pooling mode: {pool_mode}")

    if aux_output and aux_sparsity_only:
        print("Auxiliary output is enabled (sparsity only)")
    elif aux_output and not aux_sparsity_only:
        print(
            "\033[31mWARNING: Including all auxiliary outputs can be slow, only for analysis and debugging!\033[0m"
        )
    else:
        print("Auxiliary output is disabled")

    print("-" * 100)

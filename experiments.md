# All model defaults

- Full dimensional vectors
- Num Heads allocated
- Attention is exact (except for masking) -> Has W_v
- MLP dimensions allocated
- Attn -> MLP

# What to vary

- MLP shared or not
- Dimension of vectors
- MLP at input (for covariates only)
- MLP at output (for functional update only)
- Num heads allocated to each part
- Update targets

# grouped-query-attention-pytorch

(Unofficial) PyTorch implementation of grouped-query attention (GQA) from [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245.pdf)

Includes:
- [x] scaled dot-product attention with GQA support
- [x] GQA multi-head attention layer
- [x] Prototype (untrained) GQA Transformer models: `GQATransformer`, `GQATransformerLM`
- [x] Code to convert pretrained T5 model to use GQA
- [x] Reproduce runtime benchmarks from [GQA paper](https://arxiv.org/pdf/2305.13245.pdf) (Figure 6)
    - For more details, see [scripts/README.md](scripts/README.md)

To do:
- [ ] Fine-tuning code for T5 GQA models
- [ ] Reproduce fine-tuning results from [GQA paper](https://arxiv.org/pdf/2305.13245.pdf) (Figures 3,5)

## Install

PyPI:
```bash
pip install grouped-query-attention-pytorch
```

From source:
```bash
pip install "grouped-query-attention-pytorch @ git+ssh://git@github.com/fkodom/grouped-query-attention-pytorch.git"
```

For contributors:
```bash
# Install all dev dependencies (tests, T5 support, etc.)
pip install "grouped-query-attention-pytorch[test,t5] @ git+ssh://git@github.com/fkodom/grouped-query-attention-pytorch.git"
# Setup pre-commit hooks
pre-commit install
```


## Benchmark

I attempt to reproduce the runtime benchmarks from the [GQA paper](https://arxiv.org/pdf/2305.13245.pdf) (Figure 6).  Unfortunately, I don't have access to the same hardware, so the comparison isn't perfect. (They use multiple high-end GPUs, and I use a single 2080 Ti.)  Even with different hardware, though, it is clear that runtime scales similarly with the number of GQA groups.

For more details, see [scripts/README.md](scripts/README.md)

> Left: This repo <br> Right: Original paper
<p float="left">
    <img src="doc/benchmark_t5.png" alt="drawing" width="400"/>
    <img src="doc/benchmark_t5_original.png" alt="drawing" width="400"/>
</p>
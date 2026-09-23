# The Assay model family

Every row is the same recipe on a different backbone, evaluated on the same splits. Cells are
accuracy / Brier / ECE after one temperature fitted on seen-task calibration data. Unseen
tasks are eleven datasets never trained on; the transfer suite is `jaredpalmer/kev-suites`
transfer-v4 dev, whose sources are excluded from training. Abstention is the fitted conformal
act rate and the error among answers acted on, at alpha 0.1, measured on unseen tasks.
Latency is one question / 24 packed questions over one state.

| model | backbone | size | seen (dev) | unseen (holdout) | transfer-v4 | abstention (unseen) | latency |
|---|---|---|---|---|---|---|---|
| [assay-0.6b](https://huggingface.co/Berk/assay-0.6b) | Qwen3-0.6B-Base | 0.6B | 0.705 / 0.391 / 0.030 | 0.704 / 0.397 / 0.037 | 0.636 / 0.499 / 0.124 | bool 49% at 14.8%, choice 39% at 5.6% | 17.4 / 21.8 ms (GPU) |
| [assay-1.7b](https://huggingface.co/Berk/assay-1.7b) | Qwen3-1.7B-Base | 1.7B | 0.740 / 0.355 / 0.035 | 0.752 / 0.334 / 0.024 | 0.670 / 0.436 / 0.115 | bool 68% at 11.6%, choice 53% at 6.3% | 17.8 / 32.1 ms (GPU) |
| [assay-4b](https://huggingface.co/Berk/assay-4b) | Qwen3-4B-Base | 4B | 0.791 / 0.287 / 0.027 | 0.803 / 0.271 / 0.023 | 0.784 / 0.302 / 0.061 | bool 75% at 10.2%, choice 74% at 8.0% | 22.9 / 56.5 ms (GPU) |
| [assay-8b](https://huggingface.co/Berk/assay-8b) | Qwen3-8B-Base | 8B | 0.808 / 0.266 / 0.024 | 0.808 / 0.256 / 0.021 | 0.818 / 0.267 / 0.048 | bool 75% at 8.7%, choice 78% at 8.0% | 23.2 / 84.5 ms (GPU) |
| [assay-27b](https://huggingface.co/Berk/assay-27b) | Qwen3.8-27B (4-bit) | 27B | 0.834 / 0.243 / 0.040 | 0.842 / 0.221 / 0.040 | 0.842 / 0.229 / 0.041 | bool 88% at 11.2%, choice 84% at 7.4% | 109.3 / 431.3 ms (GPU) |
| [assay-compiled-base](https://huggingface.co/Berk/assay-compiled-base) | gte-modernbert-base | 149M | 0.668 / 0.442 / 0.037 | 0.606 / 0.494 / 0.061 | 0.542 / 0.572 / 0.095 | bool 8% at 5.0% | 28.95 / 30.64 ms (CPU) |

The models and the dataset are collected at [Assay on the Hub](https://huggingface.co/collections/Berk/assay-calibrated-typed-decisions-6ab2fcb2b7eea0b7aaf785ab).

All weights are Apache-2.0. Training data: 55 public datasets rendered as typed questions
plus our own generators, published at
[Berk/assay-synthetic](https://huggingface.co/datasets/Berk/assay-synthetic).

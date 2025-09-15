# Awesome-Efficient-LLM

# Toxonomy and Papers
- [Sparsity and Pruning](#Sparsity-and-Pruning)
- [Quantization](#Quantization)
  - [LLM Quantization](#LLM-Quantization)
  - [VLM Quantization](#VLM-Quantization)
- [Knowledge Distillation](#Knowledge-Distillation)
- [Low-Rank Decomposition](#Low-Rank-Decomposition)
- [KV Cache Compression](#KV-Cache-Compression)
- [Speculative Decoding](#Speculative-Decoding)

---
# Sparsity and Pruning
| Year | Title                                                                   | Venue   | Paper                                 | code                                                                                                                        |
| ---- | ----------------------------------------------------------------------- | ------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 2023 | SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot | ICML 2023| [Link](https://arxiv.org/pdf/2301.00774) |         [Link](https://github.com/IST-DASLab/sparsegpt) ![](https://img.shields.io/github/stars/IST-DASLab/sparsegpt.svg?style=social) |

---
# Quantization
## LLM Quantization
| Year | Title                                                                   | Venue   | Paper                                 | code                                                                                                                        |
| ---- | ----------------------------------------------------------------------- | :------ | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 2023 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | ICLR 2023| [Link](https://arxiv.org/abs/2210.17323) |         [Link](https://github.com/IST-DASLab/gptq) ![](https://img.shields.io/github/stars/IST-DASLab/gptq.svg?style=social) |
| 2025 | OSTQuant: Refining Large Language Model Quantization with <br/>Orthogonal and Scaling Transformations for Better Distribution Fitting | ICLR 2025 | [Link](https://arxiv.org/pdf/2501.13987) | [Link](https://github.com/BrotherHappy/OSTQuant) ![](https://img.shields.io/github/stars/BrotherHappy/OSTQuant.svg?style=social) |
| 2025 | SpinQuant: LLM quantization with learned rotations | ICLR 2025 | [Link](https://arxiv.org/pdf/2405.16406) | [Link](https://github.com/facebookresearch/SpinQuant) ![](https://img.shields.io/github/stars/facebookresearch/SpinQuant.svg?style=social) |
| 2022 | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | ICML 2023 | [Link](https://arxiv.org/abs/2211.10438) | [Link](https://github.com/mit-han-lab/smoothquant) ![](https://img.shields.io/github/stars/mit-han-lab/smoothquant.svg?style=social) |
| 2023 | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | MLSys 2024 | [Link](https://arxiv.org/abs/2306.00978) | [Link](https://github.com/mit-han-lab/llm-awq) ![](https://img.shields.io/github/stars/mit-han-lab/llm-awq.svg?style=social) |
| 2024 | QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks | ICML 2024 | [Link](https://arxiv.org/abs/2402.04396) | [Link](https://github.com/Cornell-RelaxML/quip-sharp) ![](https://img.shields.io/github/stars/Cornell-RelaxML/quip-sharp.svg?style=social) |

## VLM Quantization

## DiT Quantization

| Year | Title | Venue | Task | Paper | Code |
|------|-------|-------|------|-------|------|
| 2025 | SVDQuant: Absorbing Outliers by Low-Rank Component for 4-Bit Diffusion Models | ICLR 2025 | T2I | [Link](https://arxiv.org/pdf/2411.05007) | [Link](https://github.com/nunchaku-tech/nunchaku) ![](https://img.shields.io/github/stars/nunchaku-tech/nunchaku.svg?style=social) |
| 2025 | ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation | ICLR 2025 | Image Generation | [Link](https://arxiv.org/pdf/2406.02540) | [Link](https://github.com/thu-nics/ViDiT-Q) ![](https://img.shields.io/github/stars/thu-nics/ViDiT-Q.svg?style=social) |
| 2023 | Post-training Quantization on Diffusion Models | CVPR 2023 | T2I、T2V | [Link](https://arxiv.org/pdf/2211.15736) | [Link](https://github.com/42Shawn/PTQ4DM) ![](https://img.shields.io/github/stars/42Shawn/PTQ4DM.svg?style=social) |
| 2023 | Q-Diffusion: Quantizing Diffusion Models | ICCV 2023 | Image Generation | [Link](https://arxiv.org/pdf/2302.04304) | [Link](https://github.com/Xiuyu-Li/q-diffusion) ![](https://img.shields.io/github/stars/Xiuyu-Li/q-diffusion.svg?style=social) |

---
# Knowledge Distillation
| Year | Title                                                                   | Venue   | Paper                                 | code                                                                                                                        |
| ---- | ----------------------------------------------------------------------- | ------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 2025 | LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation              | ICLR 2025 | [Link](https://arxiv.org/pdf/2408.15881) | [Link](https://github.com/shufangxun/LLaVA-MoD) ![](https://img.shields.io/github/stars/shufangxun/LLaVA-MoD.svg?style=social) |
---
# Low-Rank Decomposition

---
# KV Cache Compression
## Token Eviction
| Year | Title                                                        | Venue        | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | ------------ | ---------------------------------------- | ------------------------------------------------------------ |
| 2023 | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models | NeurIPS 2023 | [Link](https://arxiv.org/abs/2306.14048) | [Link](https://github.com/FMInference/H2O) ![](https://img.shields.io/github/stars/FMInference/H2O.svg?style=social) |
| 2023 | Efficient Streaming Language Models with Attention Sinks | ICLR 2024 | [Link](https://arxiv.org/abs/2309.17453) | [Link](https://github.com/mit-han-lab/streaming-llm) ![](https://img.shields.io/github/stars/mit-han-lab/streaming-llm.svg?style=social) |

## KV Cache Quantization
| Year | Title                                                        | Venue    | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | -------- | ---------------------------------------- | ------------------------------------------------------------ |
| 2024 | IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact | ACL 2024 | [Link](https://arxiv.org/abs/2403.01241) | [Link](https://github.com/ruikangliu/IntactKV) ![](https://img.shields.io/github/stars/ruikangliu/IntactKV.svg?style=social) |
| 2024 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache | ICML 2024 | [Link](https://arxiv.org/abs/2402.02750) | [Link](https://github.com/jy-yuan/KIVI) ![](https://img.shields.io/github/stars/jy-yuan/KIVI.svg?style=social) |
| 2024 | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization | NeurIPS 2024 | [Link](https://arxiv.org/abs/2401.18079) | [Link](https://github.com/SqueezeAILab/KVQuant) ![](https://img.shields.io/github/stars/SqueezeAILab/KVQuant.svg?style=social) |


---
# Speculative Decoding


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
| 2023 | GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | ICLR   2023 | [Link](https://arxiv.org/abs/2210.17323) |         [Link](https://github.com/IST-DASLab/gptq) ![](https://img.shields.io/github/stars/IST-DASLab/gptq.svg?style=social) |
| 2025 | OSTQuant: Refining Large Language Model Quantization with <br/>Orthogonal and Scaling Transformations for Better Distribution Fitting | ICLR   2025 | [Link](https://arxiv.org/abs/2405.16406) | [Link](https://github.com/BrotherHappy/OSTQuant) ![](https://img.shields.io/github/stars/BrotherHappy/OSTQuant.svg?style=social) |
| 2025 | SpinQuant: LLM quantization with learned rotations | ICLR   2025 | [Link](https://arxiv.org/abs/2501.13987) | [Link](https://github.com/facebookresearch/SpinQuant) ![](https://img.shields.io/github/stars/facebookresearch/SpinQuant.svg?style=social) |
| 2022 | SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | ICML  2023 | [Link](https://arxiv.org/abs/2211.10438) | [Link](https://github.com/mit-han-lab/smoothquant) ![](https://img.shields.io/github/stars/mit-han-lab/smoothquant.svg?style=social) |
| 2023 | AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | MLSys 2024 | [Link](https://arxiv.org/abs/2306.00978) | [Link](https://github.com/mit-han-lab/llm-awq) ![](https://img.shields.io/github/stars/mit-han-lab/llm-awq.svg?style=social) |
| 2024 | QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks | ICML  2024 | [Link](https://arxiv.org/abs/2402.04396) | [Link](https://github.com/Cornell-RelaxML/quip-sharp) ![](https://img.shields.io/github/stars/Cornell-RelaxML/quip-sharp.svg?style=social) |
| 2025 | QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving | MLSys 2025 | [Link](https://arxiv.org/abs/2405.04532) |  [Link](https://github.com/mit-han-lab/omniserve) ![](https://img.shields.io/github/stars/mit-han-lab/omniserve.svg?style=social) |
| 2024 | QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs         | NeurIPS 2024 | [Link](https://arxiv.org/abs/2404.00456) |[Link](https://github.com/spcl/QuaRot)![](https://img.shields.io/github/stars/spcl/QuaRot.svg?style=social)|
| 2024 | Atom: Low-bit Quantization for Efficient and Accurate LLM Serving | MLSys 2024 | [Link](https://arxiv.org/abs/2404.00456) |[Link](https://github.com/efeslab/Atom)![](https://img.shields.io/github/stars/efeslab/Atom.svg?style=social) |



## VLM Quantization

---
# Knowledge Distillation
| Year | Title                                                                   | Venue   | Paper                                 | code                                                                                                                        |
| ---- | ----------------------------------------------------------------------- | ------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 2025 | LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation              | ICLR 2025 | [Link](https://arxiv.org/pdf/2408.15881) | [Link](https://github.com/shufangxun/LLaVA-MoD) ![](https://img.shields.io/github/stars/shufangxun/LLaVA-MoD.svg?style=social) |
---
# Low-Rank Decomposition
| Year | Title                                                                   | Venue   | Paper                                 | code                                                                                                                        |
| ---- | ----------------------------------------------------------------------- | ------- | ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| 2024 | Compressing Large Language Models using Low Rank and Low Precision Decomposition | NeurIPS 2024 | [Link](https://openreview.net/pdf?id=lkx3OpcqSZ) | [Link](https://github.com/pilancilab/caldera) ![](https://img.shields.io/github/stars/pilancilab/caldera.svg?style=social) |
| 2022 | Compressible-composable NeRF via Rank-residual Decomposition | NeurIPS 2022 | [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/5ed5c3c846f684a54975ad7a2525199f-Paper-Conference.pdf) | [Link](https://github.com/ashawkey/CCNeRF) ![](https://img.shields.io/github/stars/ashawkey/CCNeRF.svg?style=social) |
| 2024 | Unified Low-rank Compression Framework for Click-through Rate Prediction | KDD 2024 | [Link](https://arxiv.org/abs/2405.18146) | [Link](https://github.com/yuhao318/Atomic_Feature_Mimicking) ![](https://img.shields.io/github/stars/yuhao318/Atomic_Feature_Mimicking.svg?style=social) |
| 2025 | Pivoting Factorization: A Compact Meta Low-Rank Representation of Sparsity for Efficient Inference in Large Language Models | ICML 2025 | [Link](https://icml.cc/virtual/2025/poster/46433) | [Link](https://github.com/biomedical-cybernetics/pivoting-factorization) ![](https://img.shields.io/github/stars/biomedical-cybernetics/pivoting-factorization.svg?style=social) |
| 2024 | SliceGPT: Orthogonal Slicing for Parameter-Efficient Transformer Compression | ICLR 2024 | [Link](https://arxiv.org/abs/2401.15024) | [Link](https://github.com/microsoft/TransformerCompression) ![](https://img.shields.io/github/stars/microsoft/TransformerCompression.svg?style=social) |
---
# KV Cache Compression
## Token Eviction(also known as Token Selection)
| Year | Title                                                        | Venue        | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | ------------ | ---------------------------------------- | ------------------------------------------------------------ |
| 2023 | H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models | NeurIPS 2023 | [Link](https://arxiv.org/abs/2306.14048) | [Link](https://github.com/FMInference/H2O) ![](https://img.shields.io/github/stars/FMInference/H2O.svg?style=social) |
| 2023 | Efficient Streaming Language Models with Attention Sinks | ICLR 2024 | [Link](https://arxiv.org/abs/2309.17453) | [Link](https://github.com/mit-han-lab/streaming-llm) ![](https://img.shields.io/github/stars/mit-han-lab/streaming-llm.svg?style=social) |
| 2024 | SnapKV: LLM Knows What You are Looking for Before Generation | NeurIPS 2024 | [Link](https://arxiv.org/abs/2404.14469) | [Link](https://github.com/FasterDecoding/SnapKV) ![](https://img.shields.io/github/stars/FasterDecoding/SnapKV.svg?style=social) |
| 2025 | R-KV: Redundancy-aware KV Cache Compression for Reasoning Models |  | [Link](https://arxiv.org/abs/2505.24133) | [Link](https://github.com/Zefan-Cai/R-KV) ![](https://img.shields.io/github/stars/Zefan-Cai/R-KV.svg?style=social) |

## Budget Allocation
| Year | Title                                                        | Venue    | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | -------- | ---------------------------------------- | ------------------------------------------------------------ |
| 2024 | PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling |  | [Link](https://arxiv.org/abs/2406.02069v4) | [Link](https://github.com/Zefan-Cai/KVCache-Factory) ![](https://img.shields.io/github/stars/Zefan-Cai/KVCache-Factory.svg?style=social) |
| 2025 | LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models | ICML 2025 | [Link](https://arxiv.org/abs/2507.14204) | [Link](https://github.com/GATECH-EIC/LaCache) ![](https://img.shields.io/github/stars/GATECH-EIC/LaCache.svg?style=social) |
| 2025 | CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences | ICLR 2025 | [Link](https://arxiv.org/abs/2503.12491) | [Link](https://github.com/antgroup/cakekv) ![](https://img.shields.io/github/stars/antgroup/cakekv.svg?style=social) |

## KV Cache Merging
| Year | Title                                                        | Venue    | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | -------- | ---------------------------------------- | ------------------------------------------------------------ |
| 2024 | MiniCache: KV Cache Compression in Depth Dimension for Large Language Models | NeurIPS 2024 | [Link](https://arxiv.org/abs/2405.14366) | [Link](https://github.com/AkideLiu/MiniCache) ![](https://img.shields.io/github/stars/AkideLiu/MiniCache.svg?style=social) |
| 2024 | CaM: Cache Merging for Memory-efficient LLMs Inference | ICML 2024 | [Link](https://openreview.net/forum?id=LCTmppB165) | [Link](https://github.com/zyxxmu/cam) ![](https://img.shields.io/github/stars/zyxxmu/cam.svg?style=social) |
| 2024 | D2O: Dynamic Discriminative Operations for Efficient Long-Context Inference of Large Language Models | ICLR 2025 | [Link](https://arxiv.org/abs/2406.13035) | [Link](https://github.com/AIoT-MLSys-Lab/D2O) ![](https://img.shields.io/github/stars/AIoT-MLSys-Lab/D2O.svg?style=social) |

## KV Cache Quantization
| Year | Title                                                        | Venue    | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | -------- | ---------------------------------------- | ------------------------------------------------------------ |
| 2024 | IntactKV: Improving Large Language Model Quantization by Keeping Pivot Tokens Intact | ACL 2024 | [Link](https://arxiv.org/abs/2403.01241) | [Link](https://github.com/ruikangliu/IntactKV) ![](https://img.shields.io/github/stars/ruikangliu/IntactKV.svg?style=social) |
| 2024 | KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache | ICML 2024 | [Link](https://arxiv.org/abs/2402.02750) | [Link](https://github.com/jy-yuan/KIVI) ![](https://img.shields.io/github/stars/jy-yuan/KIVI.svg?style=social) |
| 2024 | KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization | NeurIPS 2024 | [Link](https://arxiv.org/abs/2401.18079) | [Link](https://github.com/SqueezeAILab/KVQuant) ![](https://img.shields.io/github/stars/SqueezeAILab/KVQuant.svg?style=social) |
| 2024 | SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models | COLM 2024 | [Link](https://arxiv.org/abs/2405.06219) | [Link](https://github.com/cat538/SKVQ) ![](https://img.shields.io/github/stars/cat538/SKVQ.svg?style=social) |



---
# Speculative Decoding
| Year | Title                                                        | Venue    | Paper                                    | code                                                         |
| ---- | ------------------------------------------------------------ | -------- | ---------------------------------------- | ------------------------------------------------------------ |
| 2024 | Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting|NeurIPS 2024|[Kangaroo](https://arxiv.org/abs/2404.18911) | [code](https://github.com/Equationliu/Kangaroo) ![](https://img.shields.io/github/stars/Equationliu/Kangaroo.svg?style=social)|
| 2024 | EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees|EMNLP 2024|[EAGLE2](https://arxiv.org/abs/2406.16858) | [code](https://github.com/SafeAILab/EAGLE) ![](https://img.shields.io/github/stars/SafeAILab/EAGLE.svg?style=social)|
| 2025 |Learning Harmonized Representations for Speculative Sampling|ICLR 2025|[HASS](https://arxiv.org/pdf/2408.15766) |[code](https://github.com/HArmonizedSS/HASS) ![](https://img.shields.io/github/stars/HArmonizedSS/HASS.svg?style=social)|
| 2025 |Parallel Speculative Decoding with Adaptive Draft Length|ICLR 2025|[PEARL](https://arxiv.org/pdf/2408.11850) |[code](https://github.com/smart-lty/ParallelSpeculativeDecoding) ![](https://img.shields.io/github/stars/smart-lty/ParallelSpeculativeDecoding.svg?style=social)|




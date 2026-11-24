# Hi, I'm Weixi Lin

I'm a computer technology master's student at Northwestern Polytechnical University (NWPU). I work on distributed training, efficient inference, and reinforcement learning post-training. I'm especially interested in how models share GPU memory and how to keep training and inference running reliably.

I'm a [verl-omni committer](https://github.com/verl-project/verl-omni/blob/main/docs/community/governance.md#active-committers), working on the trainer, rollout and agent loop, and reward system. I'm open to internships in LLM inference and AI systems, remote or on-site. [Email me](mailto:weixilin307@gmail.com).

## Open source

### What I've shipped

- **verl-omni:** I added opt-in [FSDP2 sharding for PickScore](https://github.com/verl-project/verl-omni/pull/855) on two GPUs, [CPU placement for native rewards](https://github.com/verl-project/verl-omni/pull/694), and [named rewards for the SD3.5 OCR recipe](https://github.com/verl-project/verl-omni/pull/695).
- **vLLM-Omni and SGLang-Omni:** I fixed a [Qwen3-Omni audio dtype issue](https://github.com/vllm-project/vllm-omni/pull/8327), added [remote SeedVR2 restoration in ComfyUI](https://github.com/vllm-project/vllm-omni/pull/8505), and made [PersonaPlex update caller history incrementally](https://github.com/sgl-project/sglang-omni/pull/3103).
- **UniRL and llm-d-async:** I made [Wan2.1 denoising noise reproducible across rollout microbatches](https://github.com/Tencent-Hunyuan/UniRL/pull/652) and updated the [Kubernetes inference gateway dependency and tests](https://github.com/llm-d/llm-d-async/pull/558).

### What I'm working on

- **Multimodal RL:** Making [diffusion rollout and training take turns on shared GPUs](https://github.com/verl-project/verl-omni/pull/742) in verl-omni, and exploring [how to resume interrupted denoising under the original policy](https://github.com/vllm-project/vllm-omni/issues/10529) in vLLM-Omni.
- **Inference systems:** Working on [multimodal prefix reuse](https://github.com/vllm-project/vllm/pull/56818) in vLLM and [prefill–decode separation](https://github.com/sgl-project/sglang-omni/pull/4388) in SGLang-Omni.
- **Training efficiency:** Working on [GDN activation offloading](https://github.com/NVIDIA/Megatron-LM/pull/7908) and [partial CUDA Graph capture with FSDP v2](https://github.com/NVIDIA/Megatron-LM/pull/9112) in Megatron-LM.

<details>
<summary>More ongoing work and related PRs</summary>

- **verl-omni:** [Per-sample reward streaming](https://github.com/verl-project/verl-omni/pull/730), [audio inputs for shared reward managers](https://github.com/verl-project/verl-omni/pull/920), and [safe rollout recovery after validation scoring](https://github.com/verl-project/verl-omni/pull/929).
- **vLLM-Omni:** [Configured pipeline transitions](https://github.com/vllm-project/vllm-omni/pull/8745), [SD3 step execution](https://github.com/vllm-project/vllm-omni/pull/10864), [LoRA export and restore](https://github.com/vllm-project/vllm-omni/pull/10938), and [denoising suspension and recovery](https://github.com/vllm-project/vllm-omni/pull/10942).
- **SGLang-Omni:** [PersonaPlex warmup](https://github.com/sgl-project/sglang-omni/pull/3104), [bounded audio batching](https://github.com/sgl-project/sglang-omni/pull/3591), [decode KV reservation rollback](https://github.com/sgl-project/sglang-omni/pull/4384), and [KV layout validation](https://github.com/sgl-project/sglang-omni/pull/4392).
- **Megatron-LM and Transformer Engine:** [Activation offload tests](https://github.com/NVIDIA/Megatron-LM/pull/8243), [DCP scheduling metrics](https://github.com/NVIDIA/Megatron-LM/pull/8431), and [CUDA Graph warmup cleanup](https://github.com/NVIDIA/TransformerEngine/pull/3862).
- **NVIDIA Model Optimizer and Megatron-LM:** [EAGLE3 teacher-feature streaming proposal](https://github.com/NVIDIA/Model-Optimizer/issues/3474) and draft implementations in [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer/pull/3473) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/pull/9550).
- **Mooncake and llm-d:** [Encoder–Prefill–Decode disaggregation and agent state reuse](https://github.com/kvcache-ai/Mooncake/pull/2836) and a [shared latency feature schema proposal](https://github.com/llm-d/llm-d-router/issues/4218).
- **UniRL and NVIDIA recsys-examples:** [Wan2.1 on-policy distillation recipe](https://github.com/Tencent-Hunyuan/UniRL/pull/651) and [DynamicEmb row-index resolution proposal](https://github.com/NVIDIA/recsys-examples/issues/552) ([draft implementation](https://github.com/NVIDIA/recsys-examples/pull/551)).

</details>

[All public upstream pull requests](https://github.com/search?q=author%3AWEIXILIN0307+is%3Apr+-user%3AWEIXILIN0307+is%3Apublic&type=pullrequests)

## Selected projects

- **[MiniCPM-o on Ascend](https://github.com/WEIXILIN0307/llama.cpp-omni/tree/feat/ascend-cann):** Optimized multimodal inference across two NPUs, including CPU/NPU overlap, CANN operators, graph execution, and W8A8 quantization.
- **[Efficient VLM inference](https://github.com/WEIXILIN0307/AICASGC):** Optimized Qwen3-VL-2B with KV cache reuse, projection fusion, Triton kernels, and CUDA Graphs.
- **[MetaX C500 kernels](https://github.com/WEIXILIN0307/TileOPs-Metax/tree/feat/quant-per-channel-cast-fused):** Built and tuned fused quantization kernels with TileLang, profiling, and correctness tests.

I mostly work with Python, PyTorch, CUDA, Triton, TileLang, and CANN / AscendC.

## Competitions and awards

- **MiniCPM × Ascend Inference Optimization Challenge:** 5th on the public leaderboard
- **IEEE AICAS 2026 Grand Challenge:** 7th place
- **CCF Open Source Innovation Competition:** Third Prize, Mooncake KVCache
- **Huawei MindSpore Model Development Challenge:** 3rd place / Silver, S1-MoE
- **MetaX Open Source Talent Camp:** Best Engineering Practice Award

## Education

- **Northwestern Polytechnical University:** Master's in Computer Technology, 2025–present
- **Shaanxi Normal University:** Bachelor's in Software Engineering, 2021–2025

# Hi, I'm WEIXI LIN

I'm a master's student in Computer Technology at Northwestern Polytechnical University (NWPU). I work on AI infrastructure for distributed training, efficient inference, and reinforcement learning (RL) post-training. My work includes GPU/NPU kernels, KV cache management, and rollout and reward systems.

I'm looking for internships in **LLM inference optimization and AI systems**, remote or on-site.

[Email](mailto:weixilin307@gmail.com) · [GitHub](https://github.com/WEIXILIN0307)

## Open-source contributions

### Merged

- **vLLM-Omni:** Fixed a Qwen3-Omni Code2Wav dtype mismatch by preserving the embedding dtype after the batch-invariant mean, with regression coverage for BF16, FP16, and FP32 ([#8327](https://github.com/vllm-project/vllm-omni/pull/8327)).
- **SGLang-Omni:** Made PersonaPlex caller-history updates incremental on CPU and GPU, avoiding repeated full-history copies while preserving history for re-prefill ([#3103](https://github.com/sgl-project/sglang-omni/pull/3103)).
- **verl-omni:** Added opt-in FSDP2 sharding for frozen PickScore reward models across two GPUs, with lifecycle and failure cleanup handling ([#855](https://github.com/verl-project/verl-omni/pull/855)). Initial support targets single-node synchronous diffusion training.
- **UniRL:** Made WAN2.1 denoising noise reproducible by sample identity across rollout microbatches ([#652](https://github.com/Tencent-Hunyuan/UniRL/pull/652)).
- **verl-omni:** Added CPU placement for native reward models ([#694](https://github.com/verl-project/verl-omni/pull/694)) and migrated the SD3.5 OCR recipe to named rewards ([#695](https://github.com/verl-project/verl-omni/pull/695)).
- **vLLM-Omni:** Added remote SeedVR2 video restoration for ComfyUI ([#8505](https://github.com/vllm-project/vllm-omni/pull/8505)).

### In progress

- **verl-omni:** Shared-GPU asynchronous diffusion rollout ([#742](https://github.com/verl-project/verl-omni/pull/742)); per-sample named reward streaming ([#730](https://github.com/verl-project/verl-omni/pull/730)), audio inputs for shared reward managers ([#920](https://github.com/verl-project/verl-omni/pull/920)), and safe rollout restoration after synchronous validation scoring ([#929](https://github.com/verl-project/verl-omni/pull/929)).
- **Megatron-LM / Transformer Engine:** GDN recurrence activation offloading ([#7908](https://github.com/NVIDIA/Megatron-LM/pull/7908)) and regression coverage ([#8243](https://github.com/NVIDIA/Megatron-LM/pull/8243)); context-parallel scheduling metrics ([#8431](https://github.com/NVIDIA/Megatron-LM/pull/8431)); draft work on partial CUDA Graph capture with FSDP v2 and MoE ([#9112](https://github.com/NVIDIA/Megatron-LM/pull/9112)) and warmup failure cleanup ([TE #3862](https://github.com/NVIDIA/TransformerEngine/pull/3862)).
- **NVIDIA Model Optimizer / Megatron-LM:** Proposed streaming vLLM teacher features into Megatron-LM for EAGLE3 fine-tuning ([RFC #3474](https://github.com/NVIDIA/Model-Optimizer/issues/3474); draft PRs [#3473](https://github.com/NVIDIA/Model-Optimizer/pull/3473) and [#9550](https://github.com/NVIDIA/Megatron-LM/pull/9550)).
- **vLLM / vLLM-Omni:** Multimodal prefix-cache boundary handling for sibling requests on hybrid Mamba models ([vLLM #56818](https://github.com/vllm-project/vllm/pull/56818)); configurable pipeline stage transitions ([Omni #8745](https://github.com/vllm-project/vllm-omni/pull/8745)); proposed same-policy diffusion request recovery ([Omni RFC #10529](https://github.com/vllm-project/vllm-omni/issues/10529)).
- **SGLang-Omni:** PersonaPlex session LM warmup before server readiness ([#3104](https://github.com/sgl-project/sglang-omni/pull/3104)) and bounded batching of queued session audio ([#3591](https://github.com/sgl-project/sglang-omni/pull/3591)).
- **Mooncake / llm-d:** Encoder–Prefill–Decode disaggregation and reusable agent state ([Mooncake #2836](https://github.com/kvcache-ai/Mooncake/pull/2836)); proposed a shared latency feature schema across llm-d's router and predictor ([llm-d RFC #4218](https://github.com/llm-d/llm-d-router/issues/4218)).
- **UniRL / NVIDIA recsys-examples:** Wan2.1 on-policy distillation recipe ([UniRL #651](https://github.com/Tencent-Hunyuan/UniRL/pull/651)); proposed receiver-side DynamicEmb row-index resolution during input distribution ([RFC #552](https://github.com/NVIDIA/recsys-examples/issues/552), [draft prototype #551](https://github.com/NVIDIA/recsys-examples/pull/551)).

[View my upstream pull requests](https://github.com/search?q=author%3AWEIXILIN0307+is%3Apr+-user%3AWEIXILIN0307+is%3Apublic&type=pullrequests)

## Selected projects

- **[MiniCPM-o on Ascend](https://github.com/WEIXILIN0307/llama.cpp-omni/tree/feat/ascend-cann):** Optimized multimodal inference across two NPUs, including CPU/NPU overlap, CANN operators, graph execution, and W8A8 quantization.
- **[Efficient VLM inference](https://github.com/WEIXILIN0307/AICASGC):** Optimized Qwen3-VL-2B with KV cache reuse, projection fusion, Triton kernels, and CUDA Graphs.
- **[MetaX C500 kernels](https://github.com/WEIXILIN0307/TileOPs-Metax/tree/feat/quant-per-channel-cast-fused):** Built and tuned fused quantization kernels with TileLang, profiling, and correctness tests.

## Interests and tools

- LLM inference, multimodal serving, KV and prefix caching
- Distributed training, activation offloading, and runtime scheduling
- Quantization, GPU/NPU kernels, profiling, and performance analysis

Python · PyTorch · CUDA · Triton · TileLang · CANN / AscendC · vLLM · llama.cpp

## Competitions and awards

- **MiniCPM × Ascend Inference Optimization Challenge:** 5th on the public leaderboard
- **IEEE AICAS 2026 Grand Challenge:** 7th place
- **CCF Open Source Innovation Competition:** Third Prize, Mooncake KVCache
- **Huawei MindSpore Model Development Challenge:** 3rd place / Silver, S1-MoE
- **MetaX Open Source Talent Camp:** Best Engineering Practice Award

## Education

- **Northwestern Polytechnical University:** Master's in Computer Technology, 2025–present
- **Shaanxi Normal University:** Bachelor's in Software Engineering, 2021–2025

# QwenNav vLLM Rollout Replacement Plan

本文档描述如何将当前 GenArk-QwenNav RFT 中的 HuggingFace `model.generate()` rollout 路径替换为 vLLM。目标读者是接手实现的 coding agent / research engineer。

## 1. 背景

当前 GenArk embodied training pipeline 不是使用 RLinf 已有的 text/reasoning `VLLMWorker`，而是走 embodied 专用路径：

```text
EnvWorker
→ MultiStepRolloutWorker
→ QwenNavPolicy.predict_action_batch()
→ QwenNavPolicy._batch_generate()
→ HF model.generate()
```

关键文件：

```text
examples/embodiment/train_embodied_agent.py
rlinf/workers/rollout/hf/huggingface_worker.py
rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py
```

因此，不能简单把配置里的 rollout backend 改成 `vllm`。RLinf 现有的 `rlinf/workers/rollout/vllm/vllm_worker.py` 是面向 text / reasoning `RolloutRequest → SeqGroupInfo` 的，不直接适配 embodied env loop。

正确替换点是：

```text
QwenNavPolicy._batch_generate()
```

也就是说，保持 embodied framework 不变，只替换 QwenNavPolicy 内部的 generation backend。

## 2. 当前 HF 生成路径

当前 `_batch_generate()` 位于：

```text
rlinf/models/embodiment/qwen_nav/qwen_nav_policy.py
```

核心逻辑：

```python
inputs = self.processor(
    text=sub_prompts,
    images=sub_images,
    padding=True,
    return_tensors="pt",
).to(self._model_device)

gen_out = self.model.generate(
    **inputs,
    max_new_tokens=self.max_new_tokens,
    do_sample=self.do_sample,
    temperature=self.temperature,
)
```

该函数返回：

```python
decoded: list[str]
gen_ids_list: list[torch.Tensor]  # each tensor shape: (1, response_len)
```

后续逻辑依赖这两个输出：

```text
decoded text → parse_lavira_json() → env action
response_ids → _build_forward_inputs_for_env() → actor recompute logprobs
```

因此 vLLM backend 必须保持同样接口。

## 3. 已完成 POC 结果

已新增独立 benchmark：

```text
tools/bench_qwen_nav_vllm.py
```

它不接入训练，只测试：

```text
QwenNav prompt
multi-image input
HF / vLLM generation
JSON parse success
latency
```

在 synthetic QwenNav 多图 benchmark 上结果：

```text
HF:
32 samples
latency = 79.34s
samples_per_s = 0.403
parse_ok = 31/32 = 96.88%

vLLM:
32 samples
latency = 13.02s
samples_per_s = 2.457
parse_ok = 32/32 = 100%
```

vLLM 在该 POC 中约为：

```text
2.457 / 0.403 ≈ 6.1x faster
```

注意：该结果基于 synthetic images，不等价于真实 GenArk rollout，但说明 vLLM 多图输入、QwenNav prompt、JSON parse 链路可行。

## 4. 总体设计原则

正式接入时保持以下模块不动：

```text
EnvWorker
MultiStepRolloutWorker
GenArk env
reward design
GRPO advantage
FSDP actor training
actor recompute_prev_logprobs
```

只替换 rollout policy 内部的生成层：

```text
HF model.generate()
→ vLLM LLM.generate()
```

vLLM 只负责：

```text
prompt + images → decoded text + response token ids
```

仍由 QwenNavPolicy 负责：

```text
history cache
prompt construction
parse_lavira_json
macro action buffering
forward_inputs construction
ppo_token_loss_mask construction
action distribution logger
```

仍由 actor 负责：

```text
recompute_prev_logprobs
teacher-forcing logprobs
PPO / GRPO loss
weight update
```

## 5. Phase 1: Add Generation Backend Config

在 `examples/embodiment/config/model/qwen_nav.yaml` 中新增：

```yaml
generation_backend: "hf"        # "hf" | "vllm"
vllm_gpu_memory_utilization: 0.55
vllm_tensor_parallel_size: 1
```

默认必须保持：

```yaml
generation_backend: "hf"
```

这样不会影响现有训练。

在 `QwenNavPolicy.__init__()` 中读取：

```python
self.generation_backend = str(getattr(cfg, "generation_backend", "hf"))
self.vllm_gpu_memory_utilization = float(
    getattr(cfg, "vllm_gpu_memory_utilization", 0.55)
)
self.vllm_tensor_parallel_size = int(
    getattr(cfg, "vllm_tensor_parallel_size", 1)
)
```

## 6. Phase 2: Split `_batch_generate()`

将当前 `_batch_generate()` 拆成 dispatcher + HF implementation：

```python
def _batch_generate(
    self,
    prompts: list[str],
    image_lists: list[list[Image.Image]],
) -> tuple[list[str], list[torch.Tensor]]:
    if self.generation_backend == "hf":
        return self._batch_generate_hf(prompts, image_lists)
    if self.generation_backend == "vllm":
        return self._batch_generate_vllm(prompts, image_lists)
    raise ValueError(f"Unsupported generation_backend={self.generation_backend}")
```

把原始 HF 逻辑整体移动到：

```python
def _batch_generate_hf(...):
    ...
```

该重构应当保证 `generation_backend="hf"` 时行为完全不变。

验收标准：

```text
HF training/eval still runs
HF decoded text unchanged except stochastic sampling variation
parse/action path unchanged
```

## 7. Phase 3: Add In-process vLLM Backend

新增初始化函数：

```python
def _init_vllm_engine(self):
    from vllm import LLM, SamplingParams

    self._vllm_sampling_params = SamplingParams(
        temperature=self.temperature if self.do_sample else 0.0,
        top_p=1.0,
        max_tokens=self.max_new_tokens,
    )

    llm_kwargs = {
        "model": self.model_path,
        "trust_remote_code": self._trust_remote,
        "tensor_parallel_size": self.vllm_tensor_parallel_size,
        "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
    }

    try:
        self._vllm_engine = LLM(**llm_kwargs, task="generate")
    except TypeError as exc:
        if "task" not in str(exc):
            raise
        self._vllm_engine = LLM(**llm_kwargs)
```

说明：

```text
Some vLLM versions support task="generate"; older versions do not.
The fallback is required for compatibility.
```

在 `__init__()` 中：

```python
if self.generation_backend == "vllm":
    self._init_vllm_engine()
```

短期先采用 **Mode A**：

```text
rollout worker loads both HF model and vLLM model
```

优点：

```text
smallest code change
HF fallback easy
default_forward remains available
```

缺点：

```text
rollout GPU memory doubles model residency
```

A800 80GB 上 Qwen3.5-4B 预计可先 POC。

## 8. Phase 4: Implement `_batch_generate_vllm()`

新增：

```python
@torch.inference_mode()
def _batch_generate_vllm(
    self,
    prompts: list[str],
    image_lists: list[list[Image.Image]],
) -> tuple[list[str], list[torch.Tensor]]:
    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        }
        for prompt, images in zip(prompts, image_lists, strict=True)
    ]

    outputs = self._vllm_engine.generate(
        inputs,
        sampling_params=self._vllm_sampling_params,
    )

    decoded = []
    gen_ids_list = []
    for out in outputs:
        item = out.outputs[0]
        decoded.append(item.text)
        gen_ids = torch.tensor(item.token_ids, dtype=torch.long).unsqueeze(0)
        gen_ids_list.append(gen_ids)

    return decoded, gen_ids_list
```

重要：返回格式必须与 HF implementation 完全一致：

```python
decoded: list[str]
gen_ids_list[i]: torch.Tensor shape (1, response_len)
```

这样后续无需改：

```python
fi = self._build_forward_inputs_for_env(
    single_inputs,
    resp_ids,
    decoded_text=decoded[idx],
)
```

## 9. Phase 5: Add Script Switch

在 `scripts/run_qwen_rft.sh` 中增加：

```bash
GEN_BACKEND="${GEN_BACKEND:-hf}"
```

并传入：

```bash
"actor.model.generation_backend=${GEN_BACKEND}"
"rollout.model.generation_backend=${GEN_BACKEND}"
```

运行 HF：

```bash
GEN_BACKEND=hf ./scripts/run_qwen_rft.sh ...
```

运行 vLLM：

```bash
GEN_BACKEND=vllm ./scripts/run_qwen_rft.sh ...
```

默认必须为 HF。

## 10. Phase 6: Rollout-only Validation

不要一开始就长训。先跑短配置：

```text
max_epochs=1
rollout_epoch=1
small total_num_envs
val_check_interval=-1 or very large
save_interval=large
```

重点观察：

```text
[QwenNav][action-dist]
parse_fail rate
valid/dormant samples
rollout/generate_one_epoch
actor/recompute
actor/approx_kl
actor/ratio
actor/policy_loss
actor/total_loss
NaN / inf
```

通过标准：

```text
1. rollout does not crash
2. parse_fail not worse than HF
3. action distribution not collapsed
4. actor recompute can consume vLLM response_ids
5. PPO ratio is sane
6. policy_loss / total_loss not NaN
```

## 11. Phase 7: Real GenArk Image Benchmark

Synthetic benchmark 已通过，但还需要真实图像验证。

建议新增一个小测试：

```text
collect one batch of real GenArk obs:
  main_images
  extra_view_images
  task_descriptions
  states

run QwenNavPolicy with HF
run QwenNavPolicy with vLLM
compare:
  latency
  parse_ok_rate
  action distribution
  sample decoded JSON
```

如果真实图像上仍有：

```text
parse_ok_rate comparable to HF
latency significantly lower
no action collapse
```

再进入短训练。

## 12. Critical Issue: Weight Sync

这是正式接入前最重要的风险。

当前 HF rollout worker 可以通过 RLinf 的 actor → rollout sync 路径拿到最新 actor 权重。vLLM in-process engine 默认只加载初始化 checkpoint。

如果 actor 在训练中更新，而 vLLM engine 不更新，会出现：

```text
rollout policy = old model
actor training model = updated model
```

这会导致 off-policy drift。

当前配置通常使用：

```yaml
algorithm.recompute_prev_logprobs: true
rollout.model.skip_rollout_logprobs: true
```

这可以避免 old logprobs 由 rollout engine 直接提供，但不能解决 rollout action 来自旧 policy 的问题。

因此：

```text
POC / short run:
  can ignore vLLM weight sync to validate speed and compatibility

formal training:
  must design actor → vLLM weight sync
```

可选方向：

```text
1. Periodically reload vLLM engine from checkpoint
2. Use vLLM weight loading APIs if compatible
3. Keep HF rollout for long training until sync is solved
4. For LoRA training, export/merge LoRA periodically then reload vLLM
```

Do not run long vLLM training without understanding this issue.

## 13. Mode B: Memory-optimized Rollout Worker

After Mode A works, implement Mode B.

Mode B:

```text
if generation_backend == "vllm":
    rollout worker loads processor + vLLM engine only
    does not load HF model
```

Reason:

```text
rollout worker does not need HF teacher-forcing logprobs
when skip_rollout_logprobs=True and recompute_prev_logprobs=True
```

Actor still owns HF/FSDP model for:

```text
training
recompute_prev_logprobs
teacher-forcing logprobs
```

Required refactor:

```text
QwenNavPolicy._load_model()
  should allow processor-only mode

QwenNavPolicy.default_forward()
  should remain available only in actor context

rollout worker
  should not call default_forward()
```

This is a second-stage optimization, not needed for initial integration.

## 14. Expected Speedup

Synthetic benchmark:

```text
HF:   0.403 samples/s
vLLM: 2.457 samples/s
speedup ≈ 6.1x
```

Expected real rollout:

```text
generation component: 3x-6x faster
total training step: 1.3x-2x faster
```

Total step speedup is lower because training includes:

```text
env stepping
image rendering
actor recompute
actor training
evaluation
logging
```

## 15. Main Risks

### Risk 1: vLLM environment compatibility

The current machine driver is:

```text
NVIDIA driver 535.x
CUDA driver capability 12.2
```

Very new vLLM / torch CUDA 13 builds will not work unless the driver is upgraded.

Use the already-working vLLM environment from the benchmark.

### Risk 2: Qwen3.5 model architecture support

The local checkpoint has:

```json
"model_type": "qwen3_5"
"architectures": ["Qwen3_5ForConditionalGeneration"]
"processor_class": "Qwen3VLProcessor"
```

Old vLLM / old Transformers may not recognize it.

Do not downgrade Transformers below the version required by Qwen3.5 unless using another model for POC.

### Risk 3: response token ids mismatch

vLLM token ids must be compatible with HF actor tokenizer.

If mismatch occurs:

```text
actor recompute logprobs become invalid
PPO ratio becomes meaningless
```

Mitigation:

```text
same model_path for vLLM and HF processor
sanity-check decoded text and token ids
short PPO run with ratio diagnostics
```

### Risk 4: output format regression

vLLM may change sampling behavior.

Monitor:

```text
parse_ok_rate
action distribution
stop rate
parse_fail rate
```

### Risk 5: weight sync

This is the largest formal-training risk.

POC can ignore it.

Long training cannot.

## 16. Recommended Implementation Order

```text
P1. Add generation_backend config, default hf
P2. Split _batch_generate into hf/vllm dispatcher
P3. Add in-process vLLM engine, Mode A
P4. Run HF regression test
P5. Run vLLM rollout-only test with synthetic / small real env
P6. Run 1-3 training iterations with vLLM
P7. Compare rollout/generate_one_epoch and parse/action metrics vs HF
P8. Decide whether to implement weight sync
P9. Implement Mode B processor-only rollout worker if memory is a problem
```

## 17. Acceptance Checklist

Before claiming vLLM integration is successful:

```text
[ ] generation_backend=hf remains unchanged
[ ] generation_backend=vllm initializes once per rollout worker, not every step
[ ] vLLM returns decoded text and token ids
[ ] parse_ok_rate comparable to HF
[ ] action distribution reasonable
[ ] response_ids accepted by _build_forward_inputs_for_env
[ ] actor recompute succeeds
[ ] policy_loss / total_loss not NaN
[ ] approx_kl / ratio sane
[ ] rollout/generate_one_epoch significantly lower than HF
[ ] no accidental long run without weight sync decision
```

## 18. Short Summary for Implementer

Do not replace RLinf's embodied rollout worker with the existing text `VLLMWorker`.

Instead:

```text
Implement vLLM inside QwenNavPolicy._batch_generate()
preserve the return interface
reuse all existing QwenNav parsing and forward_inputs logic
keep actor recompute path unchanged
validate with short rollout before training
```

The key engineering problem after speed POC is **actor → vLLM weight sync**.


# 复现指南

从干净的 veRL checkout 安装两项 recipe。兼容补丁提供 Qwen3-VL 的 Geo3K 提示与评分格式、验证用 presence penalty、PAPO actor 的 engine factory，以及旧 DAPO recipe 到 veRL v0 trainer 的导入适配。

## 环境

参考环境为 Linux、Python 3.12、PyTorch 2.11.0、vLLM 0.24.0、Transformers 5.10.4、Ray 2.58.0、datasets 5.0.1 与 mathruler 0.1.0。按驱动及 CUDA 选择相容的 PyTorch 与 vLLM 构建。下面的 `RELEASE_DIR` 指含 `UPSTREAM.json` 的发布目录。

```bash
export RELEASE_DIR=/path/to/math-vision
git clone https://github.com/verl-project/verl.git verl-math-vision
cd verl-math-vision
git checkout --detach 483b8a009ba3a97563edee3a19887e4862b8094a
git submodule update --init recipe
bash "$RELEASE_DIR/scripts/apply-overlay.sh" "$PWD"

uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[vllm]' mathruler pytest
```

## 数据

从 [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k)、[GeoQA 镜像](https://huggingface.co/datasets/minglingfeng/geoqa_test)、[MathVista](https://huggingface.co/datasets/AI4Math/MathVista) 和 [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) 获取数据。前三者通过 `datasets` 拉取。转换时记录源文件哈希、划分和像素重叠。重新生成的数据若与历史快照存在版本或编码差异，请将新哈希随结果保留。

```bash
mkdir -p data/geo3k
python examples/data_preprocess/geo3k.py --local_save_dir "$PWD/data/geo3k"
python "$RELEASE_DIR/scripts/fetch-eval-inputs.py" data/raw-benchmarks
python -m recipe.papo.prepare_generalization_validation
python -m recipe.moca.prepare_data \
    --train data/geo3k/train.parquet \
    --validation data/geo3k/test.parquet data/papo-eval/geoqa-test.parquet \
        data/papo-eval/mathvista-testmini.parquet data/papo-eval/mathvision-testmini.parquet \
    --output-dir data/moca
```

## 训练入口

将 `MODEL_PATH` 设为 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 的本地目录，并显式选择空闲设备。先检查 Hydra 配置，再在自己的硬件上做小批次验证。

```bash
export MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
# 比较 DAPO 与 DAPO + 遮挡对照。两组用相同 GPU 数时可共用入口。
CUDA_VISIBLE_DEVICES=0,1 PAPO_ENABLED=false \
    bash recipe/papo/run_qwen3_vl_4b_geo3k_papo_cached_2gpu.sh --cfg job --resolve
CUDA_VISIBLE_DEVICES=0,1 \
    bash recipe/papo/run_qwen3_vl_4b_geo3k_papo_cached_2gpu.sh --cfg job --resolve

# 受控训练时去掉 --cfg job --resolve。设置相同训练步数与数据。
# 只加 clean NLL 的对照保持 PAPO_ENABLED=true，设置 PAPO_PERCEPTION_COEF=0。
```

历史四卡 DAPO 基线的入口是 `examples/on_policy_distillation_trainer/run_qwen3_vl_4b_geo3k_dapo_4gpu.sh`。它沿用原有的单集训练期验证与显存参数。正式比较两个 checkpoint 时使用下面的独立四集协议。

遮挡配置使用 FSDP2、vLLM、每步保留 128 个问题、每题八次采样，默认 120 步。两卡入口开启阶段间 CPU 卸载。`perception_coef=0.01`、`clean_entropy_coef=0.03`，随机遮挡概率 0.6。名称含 `entropy` 的两项实际为采样 token NLL 代理。缓存模式的遮挡 NLL 不直接传梯度。

MoCA 的 PV/SVV 需要一个冻结的 OpenAI 兼容本地文本服务，模型名和端口由调用方设置。先核对已知正确、错误、证据缺失的行为，再进行小批次训练验收。

```bash
export MOCA_VERIFIER_URL=http://127.0.0.1:8000/v1
export MOCA_VERIFIER_MODEL=verifier
mkdir -p outputs
python -m recipe.moca.smoke_verifier \
    --base-url "$MOCA_VERIFIER_URL" --model "$MOCA_VERIFIER_MODEL" \
    --output outputs/verifier-smoke.json
MOCA_DRY_RUN=1 bash recipe/moca/run_qwen3_vl_4b_geo3k_moca_2gpu.sh
CUDA_VISIBLE_DEVICES=0,1 TOTAL_TRAINING_STEPS=2 TEST_FREQ=-1 \
    bash recipe/moca/run_qwen3_vl_4b_geo3k_moca_2gpu.sh
```

独立 MoCA 入口保留原有判分和路由设置。通过
`moca.route_only_negative_advantages=true` 可将失败轨迹的路由限于负优势 token。
PAPO 默认使用随机遮挡。通过
`actor_rollout_ref.actor.fsdp_config.papo.mask_strategy=high_attention` 或
`hybrid_attention` 可启用高分选区或按分数采样。缺少 attention 时，两种策略使用 patch 内的通道平均方差。
`mask_image_inputs` 也接受与已打包 patch 顺序一致的逐 patch `attention_weights`。
`attention_temperature` 调节采样集中度，`attention_mix_alpha` 设置均匀采样分量的权重。

## PAPO × MoCA 消融

四组共用 `data/moca` 的结构化提示、规则答案判分、GRPO 归一化和系数为 0.03 的 clean NLL。
A 只使用这些公共设置，B 加入缓存遮挡差异，C 加入证据奖励和感知 token 路由，D 同时加入两者。
四组的 masked NLL 系数均为零，以便单独估计遮挡差异项。C 和 D 的裁判只评估证据代理，最终答案使用同一规则判分。
完整设计见 [评测结果与消融](experiments.md#消融设计)。

```bash
launcher=recipe/ablation_2x2/run_qwen3_vl_4b_geo3k_2x2.sh
for arm in A B C D; do
    TRAIN_SEED=42 RUN_TAG=matched bash "$launcher" "$arm" --cfg job --resolve
done
# 去掉 --cfg job --resolve 启动训练。使用相同设备、步数和训练种子。
PAPO_PERCEPTION_COEF=0 bash "$launcher" B --cfg job --resolve
bash "$launcher" C-reward-only --cfg job --resolve
```

`PAPO_PERCEPTION_COEF=0` 生成单独命名的 B clean NLL 对照。
`C-reward-only` 使用 C 的证据奖励与原始 GRPO 优势。

## 同口径评测

保存遮挡启动命令的 `--cfg job --resolve` 输出为 `outputs/eval-config.yaml`，其中 `data.val_files` 依次指向 Geo3K、GeoQA、MathVista 和 MathVision。将 DAPO 和遮挡 checkpoint 导出成各自的 Hugging Face 模型目录。`--local_dir` 指向步骤目录下的 `actor` 子目录。

```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir /path/to/global_step_120/actor --target_dir /path/to/export

python -m recipe.papo.matched_evaluation prepare outputs/matched \
    --config outputs/eval-config.yaml --base-model "$MODEL_PATH" \
    --dapo-model /path/to/dapo/export --mask-model /path/to/mask/export
for model in base4b dapo120 papo120; do
    CUDA_VISIBLE_DEVICES=0 python -m recipe.papo.matched_evaluation evaluate \
        outputs/matched --model "$model"
done
```

每个 checkpoint、每个种子生成一份 JSONL，`acc` 是规则评分的逐题正确性。按 `data_source` 分组，先逐种子计算平均，再对三个种子取均值。`*-complete.json` 提供完整记录数。输出还保留 `length_capped`、`output_tokens`、输入哈希与模型路径，便于配对检查。

消融评测使用消融 launcher 导出的 Hydra 配置。导出各组同一步数的 checkpoint 后，按下面的方式共用评测输入和解码种子。

```bash
mkdir -p outputs
bash "$launcher" A --cfg job --resolve > outputs/ablation-eval-config.yaml
python -m recipe.papo.matched_evaluation prepare outputs/ablation-eval \
    --config outputs/ablation-eval-config.yaml --base-model "$MODEL_PATH" \
    --arm-model A=/path/to/A --arm-model B=/path/to/B \
    --arm-model C=/path/to/C --arm-model D=/path/to/D
for arm in A B C D; do
    CUDA_VISIBLE_DEVICES=0 python -m recipe.papo.matched_evaluation evaluate \
        outputs/ablation-eval --model "$arm"
done
python -m recipe.ablation_2x2.analyze_2x2_interaction \
    --a outputs/ablation-eval/A-benchmark-seed*.jsonl \
    --b outputs/ablation-eval/B-benchmark-seed*.jsonl \
    --c outputs/ablation-eval/C-benchmark-seed*.jsonl \
    --d outputs/ablation-eval/D-benchmark-seed*.jsonl \
    --seeds 42 43 44 --output outputs/ablation-interaction.json
```

分析输出按 benchmark 给出三项差值及其配对 bootstrap 95% 区间。每次重采样以题目为单位，保留同题的全部解码种子和四组预测。输出沿用输入指标的单位。
默认分析 `acc`，可用多个 `--metric` 指定逐题打分字段，例如难事实正确率和证据遗漏率。用同一评分规程为四组生成这些字段。不同训练种子的结果分别运行分析。

## CPU 检查

```bash
MOCA_TEST_TOKENIZER="$MODEL_PATH" CUDA_VISIBLE_DEVICES='' \
    python -m pytest -q recipe/papo/tests recipe/moca/tests recipe/ablation_2x2/tests
```

未配置本地 tokenizer 时，MoCA 只跳过依赖真实 tokenizer 的测试。训练、评测和裁判可能需要不同的 GPU 容量，按实际模型和输出预算做容量检查。

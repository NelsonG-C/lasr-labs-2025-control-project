#!/usr/bin/env bash
# Reproduce the closed-model causal SFT monitor arms end to end.
#
# WHY THIS EXISTS: the original OpenAI finetunes behind apps/causal
# outputs/{none,minified}_sft were deleted (the ids in SFT_ARMS 404), and the
# closed-model SFT *training* generations were never published, so those two
# conditions cannot be re-scored -- they have to be retrained from scratch.
# The resulting checkpoints are a re-run of the paper's recipe with a fresh
# SFT seed, not bit-reproductions of the original monitors.
#
# REQUIREMENTS
#   * Docker Engine running. The APPS scorer executes candidate solutions
#     against test cases in a docker sandbox, and the SFT data prep keeps only
#     examples that pass (attack mode: main_task_success and side_task_success).
#     There is no no-Docker path that preserves the paper's filtering.
#   * .env with OPENAI_API_KEY (org that will own the finetunes) and
#     GEMINI_API_KEY (H, the honeypot generator).
#   * uv, and a checkout of this repo. No GPU.
#
# USAGE
#   bash experiments/open_models/run_causal_sft_reproduction.sh [stage]
# with stage one of: generate | transform | prepare | finetune | configure |
# score | analyse | all (default). Stages are idempotent-ish and resumable:
# re-running a completed stage will refuse to clobber existing outputs, so on
# a restart pass the stage you want.
#
# After `finetune`, paste the printed model ids into SFT_ARMS in
# make_configs.py (or into experiments/open_models/ckpts.json), then continue
# with `configure`.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE="${1:-all}"
UV_WITH='--with "control-arena==2.1.3",pandas,numpy,scipy,pyyaml,dacite,click,google-genai,hf_transfer,openai,joblib,matplotlib,seaborn'
CFG=src/lasr_labs_2025_control_project/configs/apps/causal
EXP=experiments/closed-source-model/apps/causal_train
DATA=experiments/closed-source-model/apps/causal_train/sft_data

set -a; . ./.env; set +a
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-${GEMINI_API_KEY:-}}"

run_py() {
  # shellcheck disable=SC2086  # UV_WITH is a deliberate word-split arg list
  PYTHONPATH=src eval uv run --no-project $UV_WITH python "$@"
}

stage_generate() {
  echo "=== [1/7] generate causal train split (GPT-4.1 attacks, Gemini honeypots)"
  docker info >/dev/null 2>&1 || {
    echo "Docker is not running -- the APPS scorer needs it. Aborting." >&2
    exit 1
  }
  run_py src/lasr_labs_2025_control_project/scripts/generate_solutions_inspect.py \
    --config "$CFG/generator/train_generator_config.yaml"
}

stage_transform() {
  echo "=== [2/7] minify the train logs (local, no API)"
  run_py src/lasr_labs_2025_control_project/scripts/transform_solutions_inspect.py \
    --config "$CFG/transforms/minified_config_train.yaml"
}

stage_prepare() {
  echo "=== [3/7] eval logs -> raw -> prepared JSONL, per arm"
  # The monitor is trained to recognise its own (U's) generations, so the
  # positive label is U's model id; H's honeypots are the negatives.
  local target="gpt-4.1-2025-04-14"
  # none arm: untransformed solutions. minified arm: --transformed.
  for arm in none minified; do
    local logs="$DATA/$arm/eval_logs/train" raw="$DATA/$arm/raw" prep="$DATA/$arm/prepared"
    mkdir -p "$logs"
    if [ "$arm" = none ]; then
      cp "$EXP/generator/attacks.eval" "$EXP/generator/honeypots.eval" "$logs/"
      extra=""
    else
      cp "$EXP/transforms/minified/attack_transformed.eval" \
         "$EXP/transforms/minified/honeypots_transformed.eval" "$logs/"
      extra="--transformed"
    fi
    bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_logs_to_raw.sh \
      "$DATA/$arm/eval_logs" "$raw" $extra
    # --target-model is mandatory and validated: a wrong value would label
    # every example NO (both prep scripts now raise instead).
    bash src/lasr_labs_2025_control_project/finetuning/binary_tuning/transform_raw_to_prepared.sh \
      "$raw" "$target" "$prep"
  done
}

stage_finetune() {
  echo "=== [4/7] OpenAI supervised fine-tunes (2 arms; the expensive step)"
  for arm in none minified; do
    local suffix
    [ "$arm" = none ] && suffix=causal-base || suffix=causal-minified
    run_py -m lasr_labs_2025_control_project.finetuning.openai_finetune \
      --train-jsonl "$DATA/$arm/prepared/train/train_messages.jsonl" \
      --suffix "$suffix"
  done
  echo "Paste the printed ids into SFT_ARMS (causal_base / causal_minified),"
  echo "then re-run this script with the 'configure' stage."
}

stage_configure() {
  echo "=== [5/7] regenerate config trees with the new checkpoint ids"
  run_py experiments/open_models/make_configs.py --model-class closed --settings apps
  grep -q "PENDING__causal" "$CFG/monitors/causal_none_train_causal_none_eval.yaml" && {
    echo "Config still references PENDING__causal -- update SFT_ARMS first." >&2
    exit 1
  }
}

stage_score() {
  echo "=== [6/7] score the eval-side arms with the new monitors"
  for c in causal_none_train_causal_none_eval causal_minified_train_causal_minified_eval; do
    run_py src/lasr_labs_2025_control_project/scripts/monitor_solutions_inspect.py \
      --config "$CFG/monitors/$c.yaml"
  done
}

stage_analyse() {
  echo "=== [7/7] point estimates, bootstrap, plots for the two new conditions"
  for c in causal_none_train_causal_none_eval causal_minified_train_causal_minified_eval; do
    run_py src/lasr_labs_2025_control_project/scripts/plot_data.py \
      --config "$CFG/monitors/$c.yaml"
  done
  run_py src/lasr_labs_2025_control_project/scripts/bootstrap.py \
    --trees apps --replicates 1000 --out-root bootstrap_results/closed-source-model \
    --conditions apps/causal/none_sft apps/causal/minified_sft
  run_py src/lasr_labs_2025_control_project/scripts/plot_bootstrap_curves.py \
    --bootstrap-root bootstrap_results --ci 0.80
  echo "Upload: hf upload jbostock/untrusted-monitoring-2026-paper <dirs> --repo-type dataset"
}

case "$STAGE" in
  generate)  stage_generate ;;
  transform) stage_transform ;;
  prepare)   stage_prepare ;;
  finetune)  stage_finetune ;;
  configure) stage_configure ;;
  score)     stage_score ;;
  analyse)   stage_analyse ;;
  all)
    stage_generate; stage_transform; stage_prepare; stage_finetune
    echo "Stopping before 'configure': SFT_ARMS needs the new model ids first."
    ;;
  *) echo "Unknown stage: $STAGE" >&2; exit 1 ;;
esac
echo "STAGE_${STAGE}_DONE"

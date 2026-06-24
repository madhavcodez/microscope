#!/usr/bin/env bash
# MicroScope demo, walk ONE feature from activation -> label -> validation in <= 3 minutes.
#
# STATUS: the full feature-walk is GPU-bound and is enabled once Phase 1 lands on the GPU host
# (docs/PROGRESS.md Gate #1). On a CPU dev box this script verifies the toolkit installs and the
# config/determinism layer works; the gated steps below print exactly what they will run.
set -euo pipefail

echo "== MicroScope demo =="
echo "-- environment / metadata layer (CPU, always works) --"
microscope info

echo
echo "-- one-feature walk (enabled after Phase 1 on the GPU host) --"
echo "The following steps run end-to-end once microscope.{reproduce,autointerp,steering} are wired"
echo "against the verified library APIs on the GPU host. They are listed here, not yet executed:"
cat <<'STEPS'
  1) ACTIVATION : microscope reproduce --config experiments/configs/gemma2_2b_reproduce.yaml
                  (loads a pretrained Gemma Scope SAE; harvests activations for one feature)
  2) LABEL      : microscope autointerp --config experiments/configs/gemma2_2b_reproduce.yaml \
                    --n-features 1 --scorer-model <local-scorer-id>
                  (generates an explanation for the feature using the LOCAL scorer)
  3) VALIDATE   : microscope autointerp ... (detection/fuzzing/intruder score for that explanation)
                  + control: difference-of-means steering baseline for the same concept
STEPS

# To enable the real <=3-minute walk, replace this block with the three commands above once the
# GPU host is provisioned and the wrappers pass their tests.

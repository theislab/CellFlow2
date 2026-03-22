#!/bin/bash
# Embed multiple h5ad files with scConcept, scGPT, and scimilarity sequentially.
# Each model uses its own conda env; the output chains so all three
# obsm keys (X_scconcept, X_scgpt, X_scimilarity) end up in one file.

#SBATCH --job-name=emb_datasets
#SBATCH -o %x_%A_out.out
#SBATCH -e %x_%A_err.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=300G
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_80gb|h100_80gb"
#SBATCH --qos=gpu_normal
#SBATCH --time=8:00:00

DATA_DIR="/lustre/groups/ml01/workspace/xiaotong.fu/data/pancellflow/unipert"
SCRIPTS_DIR="/lustre/groups/ml01/workspace/xiaotong.fu/pancellflow/CellFlow2/scripts"
EMBED_PY="${SCRIPTS_DIR}/embed_adata.py"

INPUT_FILES=(
  # "${DATA_DIR}/tahoe_a549_w_se.h5ad"
  # "${DATA_DIR}/tahoe_hop62.h5ad"
  # "${DATA_DIR}/tahoe_aspc_1.h5ad"
  "${DATA_DIR}/tahoe_panc_1.h5ad"
  # "${DATA_DIR}/tahoe_sw48.h5ad"
  # "${DATA_DIR}/tahoe_h4.h5ad"
  # "${DATA_DIR}/tahoe_snu_1.h5ad"
  # "${DATA_DIR}/chang22.h5ad"
  # "${DATA_DIR}/mcfarland2020.h5ad"
  # "${DATA_DIR}/srivatsan20_sciplex3.h5ad"
  # "${DATA_DIR}/srivatsan20_sciplex4.h5ad"
)

source "$HOME/.bashrc"

for IN_H5AD in "${INPUT_FILES[@]}"; do
  BASENAME="$(basename "${IN_H5AD}" .h5ad)"
  OUT_H5AD="${DATA_DIR}/${BASENAME}_w_emb.h5ad"

  echo ""
  echo "######################################################################"
  echo "# ${BASENAME}"
  echo "######################################################################"


  # # ── 1. scConcept ──
  # echo "========== [${BASENAME}] scConcept =========="
  # conda deactivate
  # conda activate scconcept_env

  # python -u "${EMBED_PY}" \
  #   --in     "${IN_H5AD}" \
  #   --out    "${OUT_H5AD}" \
  #   --config "${SCRIPTS_DIR}/configs/model_scconcept.yaml"
  # echo "scConcept done (exit $?)"

  # # ── 2. scGPT ──
  # echo "========== [${BASENAME}] scGPT =========="
  # conda deactivate
  # conda activate scgpt

  # python -u "${EMBED_PY}" \
  #   --in     "${OUT_H5AD}" \
  #   --out    "${OUT_H5AD}" \
  #   --config "${SCRIPTS_DIR}/configs/model_scgpt.yaml"
  # echo "scGPT done (exit $?)"

  # ── 3. scimilarity ──
  echo "========== [${BASENAME}] scimilarity =========="
  conda deactivate
  conda activate scimilarity_env

  python -u "${EMBED_PY}" \
    --in     "${OUT_H5AD}" \
    --out    "${OUT_H5AD}" \
    --config "${SCRIPTS_DIR}/configs/model_scimilarity.yaml"
  echo "scimilarity done (exit $?)"

  echo "All embeddings written to: ${OUT_H5AD}"
done

echo ""
echo "=== ALL DONE ==="

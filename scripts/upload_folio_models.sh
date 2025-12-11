#!/bin/bash
# Upload FOLIO models to HuggingFace
# Run AFTER completing FOLIO pipeline

set -e
cd /raid/zhf004/sokrates
source venv/bin/activate

echo "============================================================"
echo "UPLOADING FOLIO MODELS TO HUGGINGFACE"
echo "============================================================"

# Check HF token
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=your_token"
    exit 1
fi

# Upload FOLIO DPO iter1
if [ -d "outputs/dpo/folio_iter1/final" ]; then
    echo "[1/3] Uploading FOLIO DPO iter1..."
    huggingface-cli upload Moonlight556/sokrates-qwen3-8b-folio-oak-dpo-iter1 \
        outputs/dpo/folio_iter1/final/ --private
fi

# Upload FOLIO DPO iter2
if [ -d "outputs/dpo/folio_iter2/final" ]; then
    echo "[2/3] Uploading FOLIO DPO iter2..."
    huggingface-cli upload Moonlight556/sokrates-qwen3-8b-folio-oak-dpo-iter2 \
        outputs/dpo/folio_iter2/final/ --private
fi

# Upload FOLIO DPO iter3
if [ -d "outputs/dpo/folio_iter3/final" ]; then
    echo "[3/3] Uploading FOLIO DPO iter3..."
    huggingface-cli upload Moonlight556/sokrates-qwen3-8b-folio-oak-dpo-iter3 \
        outputs/dpo/folio_iter3/final/ --private
fi

echo ""
echo "============================================================"
echo "FOLIO MODELS UPLOADED!"
echo "============================================================"
echo ""
echo "Models on HuggingFace:"
echo "  - Moonlight556/sokrates-qwen3-8b-folio-oak-dpo-iter1"
echo "  - Moonlight556/sokrates-qwen3-8b-folio-oak-dpo-iter2"
echo "  - Moonlight556/sokrates-qwen3-8b-folio-oak-dpo-iter3"
echo ""


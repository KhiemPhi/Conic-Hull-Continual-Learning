set -e
export HF_HUB_OFFLINE=1
echo "=== DINOv2 / COCO ==="
python -u vit_sae_conic.py --backbone dinov2 --dataset COCO   --n-img 2500 --dict 1024 --op-k 16
echo "=== CLIP / CUB200 ==="
python -u vit_sae_conic.py --backbone clip   --dataset CUB200 --n-img 3000 --dict 1024 --op-k 16
echo "=== DINOv2 / CUB200 ==="
python -u vit_sae_conic.py --backbone dinov2 --dataset CUB200 --n-img 3000 --dict 1024 --op-k 16
echo "ALL DONE"

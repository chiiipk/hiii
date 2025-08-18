# =================================================================
# === CÁC THAM SỐ BẠN CÓ THỂ THAY ĐỔI ===
# =================================================================

# 1. Đường dẫn đến mô hình SFT của bạn (quan trọng nhất!)
#    Hãy đảm bảo nó trỏ đúng đến thư mục `epoch10`.
export SFT_MODEL_DIR="./epoch10"

# 2. Tên bộ dữ liệu sở thích (các lựa chọn: 'hh', 'shp', 'se')
export DATASET_NAME="hh"

# 3. Tên thư mục output
export EXP_NAME="gpt2_kd_dpo_on_hh"

# 4. Các siêu tham số huấn luyện
export LR=2e-5
export BETA=0.1
export EPOCHS=1
export BATCH_SIZE=2 # Giảm xuống 1 nếu hết VRAM
export GRAD_ACCUMULATION=4

# =================================================================
# === LỆNH THỰC THI (không cần sửa) ===
# =================================================================

echo "Bắt đầu huấn luyện DPO cho mô hình tại: ${SFT_MODEL_DIR}"
echo "Sử dụng bộ dữ liệu: ${DATASET_NAME}"
echo "Kết quả sẽ được lưu tại thư mục có tên bắt đầu bằng: ${EXP_NAME}"

python3 -u train.py \
    model=blank_model \
    model.name_or_path=${SFT_MODEL_DIR} \
    datasets=[${DATASET_NAME}] \
    loss=dpo \
    loss.beta=${BETA} \
    exp_name=${EXP_NAME} \
    batch_size=${BATCH_SIZE} \
    eval_batch_size=16 \
    gradient_accumulation_steps=${GRAD_ACCUMULATION} \
    lr=${LR} \
    n_epochs=${EPOCHS} \
    trainer=BasicTrainer \
    eval_every=200 \
    sample_during_eval=false

echo "Hoàn thành!"
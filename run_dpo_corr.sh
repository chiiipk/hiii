
pip install -q gdown

gdown --id 1YBHFe0p6QdZFqmKjVFCVYVIiYJqZHU3O -O epoch10.zip --quiet

unzip -o epoch10.zip
ls epoch10/

export SFT_MODEL_DIR="/kaggle/working/epoch10"
export DATASET_NAME="hh"
export EXP_NAME="gpt2_kd_dpo_on_hh"

export LR=2e-5
export BETA=0.1
export EPOCHS=1
export BATCH_SIZE=16
export GRAD_ACCUMULATION=4


echo "Bắt đầu huấn luyện DPO cho mô hình tại: ${SFT_MODEL_DIR}"
echo "Sử dụng bộ dữ liệu: ${DATASET_NAME}"
echo "Kết quả sẽ được lưu tại thư mục có tên bắt đầu bằng: ${EXP_NAME}"

python3 -u /kaggle/working/hiii/train.py \
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
    eval_every=6000 \
    sample_during_eval=false

echo "Hoàn thành!"

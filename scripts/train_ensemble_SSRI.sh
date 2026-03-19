#!/bin/bash
#STATE=$1
#E0s=$2
DEVICE=$1
for STATE in 0 1 2
do
for SEED in {0..4}
do
CUDA_VISIBLE_DEVICES="$DEVICE" python run_train.py   --name="MACE_x_SS-RI_s"$STATE"_"$SEED"" \
--train_file="dataset_x/x_with_forces_0K+5000K+ci_s"$STATE"_train_v2.xyz" \
--valid_file="dataset_x/x_with_forces_0K+5000K+ci_s"$STATE"_val_v2.xyz" \
--test_file="dataset_x/x_with_forces_0K+5000K+ci_s"$STATE"_test_v2.xyz" \
--config_type_weights='{"Default":1.0}'   --E0s="average"  --model_dir='models_x2' --log_dir='logs_x2' \
--checkpoints_dir="checkpoints_x2" --model="MACE"   --hidden_irreps='128x0e + 128x1o + 128x2e'    --r_max=5.0   --batch_size=32  --max_num_epochs=500 \
--ema   --ema_decay=0.99   --amsgrad   --default_dtype="float32"   --device=cuda   --seed="$SEED"   --error_table='PerAtomMAE' --energy_key="REF_energy" \
--forces_key="REF_forces"  --eval_interval=1 --swa --patience=16 \
--lr=0.01 --swa_lr=0.001 --energy_weight=1 --forces_weight=100 --swa_energy_weight=1000 --swa_forces_weight=100 --save_cpu
done
done
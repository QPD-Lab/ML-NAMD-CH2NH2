#!/bin/bash
DEVICE=$1
# for STATE in 0 1 2
# do
for SEED in {0..4}
do
CUDA_VISIBLE_DEVICES="$DEVICE" python run_train.py   --name="MACE_x_with_forces_mst_delta_fixed_$SEED"  --config_type_weights='{"Default":1.0}'    --model_dir='models_delta' --log_dir='logs_delta' --checkpoints_dir="checkpoints_delta" --model="MACE"   --hidden_irreps='128x0e + 128x1o + 128x2e'    --r_max=5.0   --batch_size=32  --max_num_epochs=250 --ema   --ema_decay=0.99   --amsgrad   --default_dtype="float32"   --device=cuda   --seed=$SEED   --error_table='PerAtomMAE' --energy_key="REF_energy" --forces_key="REF_forces"  --eval_interval=1 --swa --patience=16 --lr=0.01 --swa_lr=0.001 --config mst_x_delta.yml
done
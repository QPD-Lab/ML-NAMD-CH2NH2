#!/bin/bash
DEVICE=$1
# for STATE in 0 1 2
# do
for SEED in {0..4}
do
CUDA_VISIBLE_DEVICES="$DEVICE" python run_train.py   --name="MACE_x_with_forces_msf_fixed_same_E0s_cas_$SEED"  --config_type_weights='{"Default":1.0}'    --model_dir='models_x' --log_dir='logs_x' --checkpoints_dir="checkpoints_x" --model="MACE"   --hidden_irreps='128x0e + 128x1o + 128x2e'    --r_max=5.0   --batch_size=32  --max_num_epochs=250 --ema   --ema_decay=0.99   --amsgrad   --default_dtype="float32"   --device=cuda   --seed=$SEED   --error_table='PerAtomMAE' --energy_key="REF_energy" --forces_key="REF_forces"  --eval_interval=1 --swa --patience=16 --lr=0.01 --swa_lr=0.001 --config=mst_x2.yml --foundation_model="models3/MACE_cas_with_forces_mst_v2_stagetwo.model" --multiheads_finetuning=True --pt_train_file="dataset_x/x_with_forces_0K+5000K+ci_s0_train_v2.xyz" --weight_pt=0
done
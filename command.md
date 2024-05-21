```
python run.py --task_name 'nodate' --is_training 1 --model_id sin --model Crossformer --data custom_withoudate --root_path './dataset/Sin/' --data_path 'sin.csv' --features S --target y --freq h --seq_len 32 --label_len 16 --pred_len 16 --inverse --enc_in 1 --dec_in 1 --c_out 1 --d_model 32 --train_epochs 1000 --patience 100
```

```1
python run.py --task_name 'nodate' --is_training 1 --model_id turb1 --model FEDformer --data custom_withoudate --root_path './dataset/LongYuanPower/processed' --data_path 'turb1.csv' --features S --target Patv --seq_len 288 --label_len 144 --pred_len 288 --inverse --enc_in 1 --dec_in 1 --c_out 1 --train_epochs 1000 --patience 5 --batch_size 32 --num_workers 1 --d_model 128 --d_ff 512 --n_heads 2 --e_layers 2 --d_layers 1
```
```
python run.py --task_name 'nodate' --is_training 1 --model_id sin --model Crossformer --data custom_withoudate --root_path './dataset/Sin/' --data_path 'sin.csv' --features S --target y --freq h --seq_len 32 --label_len 16 --pred_len 16 --inverse --enc_in 1 --dec_in 1 --c_out 1 --d_model 32 --train_epochs 1000 --patience 100
```

```1
python run.py --task_name 'nodate' --is_training 1 --model_id turb1_speed --model LSTM --data custom_withoudate --root_path './dataset/LongYuanPower/processed' --data_path 'Turb1.csv' --features MS --target Wspd --seq_len 288 --label_len 144 --pred_len 288 --enc_in 2 --dec_in 2 --c_out 1 --train_epochs 1000 --patience 6 --batch_size 32 --num_workers 0 --d_model 512 --d_ff 512 --n_heads 12 --e_layers 5 --d_layers 2 --learning_rate 0.00001
```
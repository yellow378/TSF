```
python run.py --task_name 'nodate' --is_training 1 --model_id sin --model Crossformer --data custom_withoudate --root_path './dataset/Sin/' --data_path 'sin.csv' --features S --target y --freq h --seq_len 32 --label_len 16 --pred_len 16 --inverse --enc_in 1 --dec_in 1 --c_out 1 --d_model 32 --train_epochs 1000 --patience 100
```

```1
python run.py --task_name 'long_term_forecast' --is_training 1 --model_id  Trub1_wind --model PatchTST --data custom --root_path './dataset/Wind' --data_path 'Turb1.csv' --features MS --target 'Wspd' --seq_len 720 --label_len 360 --pred_len 192 --enc_in 10 --dec_in 10 --c_out 1 --train_epochs 50 --patience 10 --batch_size 128 --num_workers 0 --d_model 512 --d_ff 512 --n_heads 8 --e_layers 6 --d_layers 5 --learning_rate 0.0001 --use_multi_gpu --devices '0,1'
```

```self Crossformer
python run.py --task_name 'long_term_forecast' --is_training 1 --model_id  Time_Future --model Crossformer --data custom --root_path './dataset/LongYuanPower/processed' --data_path 'Turb1.csv' --features MS --target 'Wspd' --seq_len 400 --label_len 0 --pred_len 400 --enc_in 10 --dec_in 10 --c_out 1 --train_epochs 10000 --patience 5 --batch_size 64 --num_workers 0 --d_model 512 --d_ff 512 --n_heads 8 --e_layers 6 --d_layers 5 --learning_rate 0.00001 --futureM 1024  --dropout 0.2  --load --use_multi_gpu  --devices '0,1'

```

```self FEDformer
python run.py --task_name 'long_term_forecast' --is_training 1 --model_id  Time_Future --model FEDformer --data custom_self --root_path './dataset/LongYuanPower/processed' --data_path 'Turb1.csv' --features MS --target 'Wspd' --seq_len 400 --label_len 100 --pred_len 400 --enc_in 10 --dec_in 10 --c_out 1 --train_epochs 10000 --patience 2 --batch_size 64 --num_workers 0 --d_model 256 --d_ff 512 --n_heads 8 --e_layers 12 --d_layers 6 --learning_rate 0.00001 --futureM 1024  --dropout 0.2  --use_multi_gpu  --devices '0,1'

```

```self Crossformer2
python run.py --task_name 'long_term_forecast' --is_training 1 --model_id  Time_Future --model Crossformer --data custom --root_path './dataset/LongYuanPower/processed' --data_path 'Turb1.csv' --features MS --target 'Wspd' --seq_len 400 --label_len 0 --pred_len 400 --enc_in 10 --dec_in 10 --c_out 1 --train_epochs 10000 --patience 5 --batch_size 16 --num_workers 0 --d_model 512 --d_ff 512 --n_heads 8 --e_layers 6 --d_layers 5 --learning_rate 0.0001 --futureM 1024  --dropout 0.1  --load --use_multi_gpu  --devices '0,1'

```
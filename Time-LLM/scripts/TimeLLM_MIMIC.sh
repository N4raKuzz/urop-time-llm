model_name=TimeLLM
train_epochs=10
learning_rate=0.02
llama_layers=8
batch_size=4
d_model=32
d_ff=128

comment='MIMIC_8_1'

python run_mimic.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/MIMIC/ \
  --data_path MIMICtable_261219.csv \
  --model_id MIMIC_8_1 \
  --model $model_name \
  --data MIMIC \
  --features M \
  --seq_len 8 \
  --pred_len 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment

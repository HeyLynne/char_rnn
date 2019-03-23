# char_rnn

Implement based on blog http://karpathy.github.io/2015/05/21/rnn-effectiveness/

#### Usage
python train.py --data_dir DIR
  --save_dir DIR
  --log_dir DIR
  --save_every 100
  --model lstm
  --rnn_size 128
  --num_layers 2
  --seq_length 50
  --batch_size 50
  --num_epochs 50
  --grad_clip 5.0
  --learning_rate 0.002
  --decay_rate 0.97
  --output_keep_prob 1.0
  --input_keep_prob 1.0

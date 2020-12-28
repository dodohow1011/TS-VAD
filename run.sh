source /home/dodohow1011/miniconda3/bin/activate py36

output_dir=tsvad_dprnn_v2_nframes40_b64
gpu=2

. ./util/parse_options.sh

python train.py -c conf/tsvad_config.json \
  -o checkpoints/$output_dir -g $gpu 

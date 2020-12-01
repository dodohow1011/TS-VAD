source /home/dodohow1011/miniconda3/bin/activate py36

output_dir=checkpoints/tsvad_nframes128b_128
gpu=0

. ./util/parse_options.sh

python train.py -c conf/tsvad_config.json \
  -o $output_dir -g $gpu

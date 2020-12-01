source /home/dodohow1011/miniconda3/bin/activate py36

output_dir=checkpoints/tsvad_nframes200_b64
gpu=1

. ./util/parse_options.sh

python train.py -c conf/tsvad_config.json \
  -o $output_dir -g $gpu

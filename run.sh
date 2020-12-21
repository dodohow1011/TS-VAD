source /home/dodohow1011/miniconda3/bin/activate py36

output_dir=tsvad_nframes30_b128
gpu=1

. ./util/parse_options.sh

python train.py -c conf/tsvad_config.json \
  -p checkpoints/tsvad_nframes40_b64/12-18_23-14_500000 \
  -o checkpoints/$output_dir -g $gpu 

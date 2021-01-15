source /home/dodohow1011/miniconda3/bin/activate py36

dset=eval
out=dprnn_v3_${dset}_460000_track1
stage=0
stop_stage=3

. ./util/parse_options.sh

if [ $stage -le 0 -a $stop_stage -ge 0 ]; then
  if [ ! -f prediction/$out/.done ]; then
    mkdir prediction/$out
    python inference.py -c conf/tsvad_config.json \
      -p checkpoints/tsvad_dprnn_v3_nframes40_b64/01-08_16-34_460000 -g 3 \
      -o prediction/$out -f data/$dset -i data/$dset

    touch prediction/$out/.done
  fi
fi
 
#TS-VAD probabilities post-processing and DER scoring

scoring=prediction/$out/scoring
hyp_rttm=$scoring/rttm
ref_rttm=chime6_rttm/${dset}_rttm
thr=0.4
window=51
min_silence=0.3
min_speech=0.2

if [ $stage -le 1 -a $stop_stage -ge 1 ]; then
  python util/convert_prob_to_rttm.py --threshold 0.4 --window 51 --min_silence 0.3 --min_speech 0.2 ark:"sort prediction/$out/weights.ark |" $hyp_rttm || exit 1;
fi

if [ $stage -le 2 -a $stop_stage -ge 2 ]; then   
  echo "Diarization results for $test"
  sed 's/_U0[1-6]\.ENH//g' $ref_rttm > $ref_rttm.scoring
  sed 's/_U0[1-6]\.ENH//g' $hyp_rttm > $hyp_rttm.scoring
  ref_rttm_path=$(readlink -f ${ref_rttm}.scoring)
  hyp_rttm_path=$(readlink -f ${hyp_rttm}.scoring)
  cd dscore && python score.py -u ../uem_file.scoring -r $ref_rttm_path \
    -s $hyp_rttm_path 2>&1 | tee -a ../$scoring/DER && cd .. || exit 1;
fi

source /home/dodohow1011/miniconda3/bin/activate py36

out=prediction
dset=dev
stage=-1
stop_stage=3

. ./util/parse_options.sh

if [ $stage -le 0 -a $stop_stage -ge 0 ]; then
  python inference.py -c conf/tsvad_config.json \
    -p checkpoints/tsvad_nframes40_b64/12-18_23-14_500000 -g 1 \
    -o $out -f data/$dset -i data/$dset
fi
 
#TS-VAD probabilities post-processing and DER scoring

scoring=$out/scoring
hyp_rttm=$scoring/rttm
ref_rttm=chime6_rttm/dev_rttm
thr=0.5
window=51
min_silence=0.3
min_speech=0.2

if [ $stage -le 1 -a $stop_stage -ge 1 ]; then
  python util/convert_prob_to_rttm.py --threshold $thr --window $window --min_silence $min_silence --min_speech $min_speech ark:"sort $out/weights.ark |" $hyp_rttm || exit 1;
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

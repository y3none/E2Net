python ./evaltools/eval.py   \
    --model  'E2Net_DINOv3' \
    --GT_root  './dataset/TestDataset' \
    --pred_root './output/Prediction/E2Net_DINOv3-test' \
    --record_path './output/Prediction/E2Net_DINOv3-test/eval_record.txt' \
    --BR 'on'
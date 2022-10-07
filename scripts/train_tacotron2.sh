mkdir -p output
#rm ./output/* && python -m multiproc train.py -m Tacotron2 --load-mel-from-disk -lr 1e-3 --epochs 100 -bs 32 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json --anneal-steps 500 1000 1500 --anneal-factor 0.1
#python -m multiproc train.py -m Tacotron2 --load-mel-from-disk --resume-from-last --epochs 100 -bs 32 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json
python -m multiproc train.py -m Tacotron2 --resume-from-last --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --log-file nvlog.json

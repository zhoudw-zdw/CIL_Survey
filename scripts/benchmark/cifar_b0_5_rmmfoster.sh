CUDA_VISIBLE_DEVICES=3  python main_rmm.py \
    -model rmm-foster \
    -init 5 \
    -incre 5 \
    -p benchmark \
    -d 0 \                     
    -m 0.2 0.3 0.3 0.3 0.3 0.6 0.6 0.5 0.5 0.5 0.5 0.5 0.6 0.6 0.6 0.6 0.6 0.5 0.6 0.6\
    -c 0.0 0.1 0.1 0.0 0.1 0.1 0.1 0.0 0.0 0.1 0.1 0.1 0.1 0.0 0.0 0.0 0.0 0.1 0.1 0.1\
    --skip
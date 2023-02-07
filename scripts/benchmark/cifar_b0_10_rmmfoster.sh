CUDA_VISIBLE_DEVICES=0 python main_rmm.py \
    -model rmm-foster \
    -init 10 \
    -incre 10 \
    -p benchmark \
    -d 0\
    -m 0.3 0.3 0.3 0.3 0.4 0.6 0.6 0.5 0.5 0.5 \
    -c 0.2 0.2 0.1 0.0 0.1 0.1 0.1 0.0 0.0 0.1 \
    --skip
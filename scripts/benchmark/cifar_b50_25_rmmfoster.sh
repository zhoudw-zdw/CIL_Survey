CUDA_VISIBLE_DEVICES=3 python main_rmm.py \
    -model rmm-foster \
    -init 50 \
    -incre 25 \
    -p benchmark \
    -d 0\
    -m 0.4 0.4  0.5 \
    -c 0.0 0.1  0.1 \
    --skip
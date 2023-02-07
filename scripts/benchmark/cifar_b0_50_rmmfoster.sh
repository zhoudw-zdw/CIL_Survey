CUDA_VISIBLE_DEVICES=0 python main_rmm.py \
    -model rmm-foster \
    -init 50 \
    -incre 50 \
    -p benchmark \
    -d 0\
    -m 0.9 0.2  \
    -c 0.2 0.1  \
    --skip
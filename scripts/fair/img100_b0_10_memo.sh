python main_memo.py \
    -model memo \
    -init 10 \
    -incre 10 \
    --dataset imagenet100 \
    -ms 2652 \
    -net memo_resnet18 \
    -p fair \
    -d 0 \
    --scheduler cosine \
    --t_max 170 \
    --train_base \
    --skip
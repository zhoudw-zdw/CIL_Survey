for number in 2000 3136 5779 7924 8574;do
CUDA_VISIBLE_DEVICES=0 python main.py \
    -model foster \
    --dataset imagenet100 \
    -net resnet18 \
    -ms $number\
    -init 50 \
    -incre 5 \
    -d 0\
    -p auc\
    --skip
done
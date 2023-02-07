for number in 2000 3136 5779 7924 8574;do
python main.py \
    -model bic \
    --dataset imagenet100 \
    -ms $number\
    -init 50 \
    -incre 5 \
    -p auc \
    -net resnet18 \
    -d 0 \
    --skip
done
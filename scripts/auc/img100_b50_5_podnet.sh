for number in 2000 3136 5779 7924 8574;do
python main.py \
    -model podnet \
    --dataset imagenet100 \
    -ms $number\
    -init 50 \
    -incre 5 \
    -p auc \
    -net cosine_resnet18 \
    -d 1
done
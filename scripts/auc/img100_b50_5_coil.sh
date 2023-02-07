for number in 2000 3136 5779 7924 8574;do
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset imagenet100 \
    -model coil \
    -ms $number\
    -init 50\
    -incre 5 \
    -net cosine_resnet18 \
    -d 0\
    -p auc
done
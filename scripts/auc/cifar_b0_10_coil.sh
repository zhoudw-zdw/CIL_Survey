for number in 2000 3634 4900 6165;do
    CUDA_VISIBLE_DEVICES=1 python main.py \
        -model coil \
        -ms $number \
        -net cosine_resnet32\
        -init 10 \
        -incre 10 \
        -p auc \
        -d 0
done
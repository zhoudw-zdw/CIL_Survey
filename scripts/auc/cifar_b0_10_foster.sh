for number in 2000 3634 4900 6165;do
CUDA_VISIBLE_DEVICES=3 python main.py \
    -model foster \
    -ms $number\
    -init 10 \
    -incre 10 \
    -d 0\
    -p auc\
    --skip
done
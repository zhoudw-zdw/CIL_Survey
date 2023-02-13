for number in 2000 3634 4900 6165;do
    python main.py \
        -model bic\
        -ms $number \
        -init 10 \
        -incre 10 \
        -p auc \
        -d 0 \
        --skip 
done
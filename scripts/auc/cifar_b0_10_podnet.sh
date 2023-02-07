for number in 2000 3634 4900 6165;do
    python main.py \
        -model podnet \
        -ms $number \
        -net cosine_resnet32\
        -init 10 \
        -incre 10 \
        -p auc \
        -d 2 
done
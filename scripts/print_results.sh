# cat $1 | grep F1 | awk '{if ($1 != "train") print $1 "\t" $6 "\t" $8 "\t" $10}' | \
# awk 'NR%2{printf "%s\t",$0;next;}1' | sed "s/,//g" | sed '/^valid/d' | \
# sort -k 8 -r | \
# python calculate_f1.py

# sudo cp $1/train.preds.txt results/score/train.preds.txt
# sudo cp $1/valid.preds.txt results/score/valid.preds.txt
# sudo cp $1/test.preds.txt results/score/test.preds.txt
sudo python3 filter.py $1/train.preds.txt > results/score/train.preds.txt
sudo python3 filter.py $1/valid.preds.txt > results/score/valid.preds.txt
sudo python3 filter.py $1/test.preds.txt > results/score/test.preds.txt
./eval.sh

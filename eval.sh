python heuristics.py results/score/train.preds.txt > results/score/train2.preds.txt
python heuristics.py results/score/valid.preds.txt > results/score/valid2.preds.txt
python heuristics.py results/score/test.preds.txt > results/score/test2.preds.txt
./conlleval < results/score/train2.preds.txt
./conlleval < results/score/valid2.preds.txt
./conlleval < results/score/test2.preds.txt

# ./conlleval < results/score/train.preds.txt
# ./conlleval < results/score/valid.preds.txt
# ./conlleval < results/score/test.preds.txt

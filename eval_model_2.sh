cd ..

./conlleval < results/score2/fold_0.preds.txt | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" - | tr -d '\n' && printf '\t'
./conlleval < results/score2/fold_1.preds.txt | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" - | tr -d '\n' && printf '\t'
./conlleval < results/score2/fold_2.preds.txt | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" - | tr -d '\n' && printf '\t'
./conlleval < results/score2/fold_3.preds.txt | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" - | tr -d '\n' && printf '\t'
./conlleval < results/score2/fold_4.preds.txt | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" - | tr -d '\n' && printf '\t'

python filter.py results/score2/fold_0.preds.txt > results/score2/fold_0f.preds.txt
python filter.py results/score2/fold_1.preds.txt > results/score2/fold_1f.preds.txt
python filter.py results/score2/fold_2.preds.txt > results/score2/fold_2f.preds.txt
python filter.py results/score2/fold_3.preds.txt > results/score2/fold_3f.preds.txt
python filter.py results/score2/fold_4.preds.txt > results/score2/fold_4f.preds.txt

./conlleval < results/score2/fold_0f.preds.txt | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" - | tr -d '\n' && printf '\t'
./conlleval < results/score2/fold_1f.preds.txt | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" - | tr -d '\n' && printf '\t'
./conlleval < results/score2/fold_2f.preds.txt | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" - | tr -d '\n' && printf '\t'
./conlleval < results/score2/fold_3f.preds.txt | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" - | tr -d '\n' && printf '\t'
./conlleval < results/score2/fold_4f.preds.txt | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" - | tr -d '\n' && printf '\t'
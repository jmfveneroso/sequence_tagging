 cat $1 | grep PER | awk '{print $3 "\t" $5 "\t" $7 "%"}' | sed "s/;//g" | tail -n 2 | paste -sd "\t" -

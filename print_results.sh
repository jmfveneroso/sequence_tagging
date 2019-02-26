cat $1 | grep F1 | awk '{if ($1 != "train") print $1 "\t" $6 "\t" $8 "\t" $10}' | awk 'NR%2{printf "%s\t",$0;next;}1' | sed "s/,//g" | sort -k 8 -r | head -n 5

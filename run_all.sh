user="$(whoami)"
if [[ $user == "root" ]]; then
  # for filename in ./configs/*.json; do
  while true; do
    ls -l ./configs/test_*.json &>/dev/null
    if [ $? -ne 0 ]; then
      break
    fi

    filename=$(ls -l ./configs/test_*.json | awk '{if (NR == 1) print $9}')
    f=${filename%.*}
    f=${f:10}
    echo "Running $filename ($f)..."
    mkdir -p results/$f
    python run.py train -j $filename > results/$f/${f}.log
    ./eval.sh >> results/$f/${f}.log

    if [ "$(ls -A checkpoints)" ]; then
      cp checkpoints/* results/$f/
    fi

    python run.py print_matrices >> results/$f/${f}.log
    if [ "$(ls -A figures)" ]; then
      mv figures/* results/$f/
    fi
    if [ "$(ls -A results/score)" ]; then
      mv results/score/* results/$f/
    fi
    mv ./configs/$f.json ./configs/$f.json.done
  done
  echo 'Done!!!'
fi

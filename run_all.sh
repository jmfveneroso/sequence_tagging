user="$(whoami)"
if [[ $user == "root" ]]; then
  for filename in ./configs/*.json; do
    f=${filename%.*}
    f=${f:10}
    echo "Running $f..."
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
  done
fi

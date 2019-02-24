import sys
import re
from os import listdir
from os.path import isfile, join

# from snippets.elmo import concat_pretrained, precalculate_elmo

# precalculate_elmo()
# concat_pretrained()

if sys.argv[1] == 'experiment':
  json = {
      "datadir": "data/ner_on_html",
      "epochs": 5,
      "batch_size": 3,
      "splitsentence": true,
      "fulldoc": true,
      "lstm_size": 100,
      "word_embeddings": "glove",
      "char_representation": "cnn",
      "decoder": "crf",
      "model": "lstm_crf"
  :

  files = [f for f in listdir('configs') if isfile(join('configs', f))]
  last_file = [(int(re.findall('[0-9]+', f)[0]), f) for f in files]
  print(last_file)

  # files = max([int(re.findall('[0-9]+', f)[0]) for f in listdir('configs') if isfile(join('configs', f))])



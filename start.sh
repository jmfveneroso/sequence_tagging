docker build -t seqtag .
docker run -d --gpus all -p 8888:8888 -v $(pwd):/code --rm --user root seqtag

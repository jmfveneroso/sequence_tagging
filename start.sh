docker build -t seqtag .
docker run -d -p 0.0.0.0:6006:6006 -p 8888:8888 -v $(pwd):/code --rm --user root seqtag

docker build -t seqtag .
docker run -d -p 8888:8888 -v $(pwd):/code --rm seqtag

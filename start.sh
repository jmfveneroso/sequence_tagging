docker build -t ner .
docker run -d -p 8888:8888 -v $(pwd):/code --rm ner

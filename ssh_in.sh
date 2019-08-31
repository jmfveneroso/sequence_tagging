docker exec -it $(docker ps | awk '{if (NR == 2) print $1}') bash

#!/bin/sh

sudo echo "$1,$2,$3" >> /home/flash/codes/people-counting/ppl-log.csv

# Convert last entry of CSV to JSON

sudo echo $(cat /home/flash/codes/people-counting/ppl-log.csv | head -n1) > /home/flash/codes/people-counting/ppl-buffer.csv
sudo echo $(cat /home/flash/codes/people-counting/ppl-log.csv | tail -n1) >> /home/flash/codes/people-counting/ppl-buffer.csv
sudo csvjson /home/flash/codes/people-counting/ppl-buffer.csv > /home/flash/codes/people-counting/ppl-payload.json

curl -v -X POST -d @/home/flash/codes/people-counting/ppl-payload.json --url http://sc-draco.flashcoffeetech.com:8080/api/v1/ruXHnIlTv4tTN30jycl1/telemetry --header "Content-Type:application/json"


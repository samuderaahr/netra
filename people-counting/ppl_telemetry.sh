#!/bin/sh

# Convert last entry of CSV to JSON

sudo echo $(cat /home/flash/codes/people-counting/log.csv | head -n1) > /home/flash/codes/people-counting/buffer.csv
sudo echo $(cat /home/flash/codes/people-counting/log.csv | tail -n1) >> /home/flash/codes/people-counting/buffer.csv
sudo csvjson /home/flash/codes/people-counting/buffer.csv > /home/flash/codes/people-counting/payload.json

curl -v -X POST -d @/home/flash/codes/people-counting/payload.json --url http://sc-draco.flashcoffeetech.com:8080/api/v1/ruXHnIlTv4tTN30jycl1/telemetry --header "Content-Type:application/json"



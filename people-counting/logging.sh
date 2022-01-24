#!/bin/sh

sudo echo "$1,$2,$3" >> /var/log/ppl-log.csv

# Convert last entry of CSV to JSON
sudo echo "{'ts':$1,'values':{'R':$2,'L':$3}}" > /var/log/ppl-payload.json 

curl -v -X POST -d @/var/log/ppl-payload.json --url $(cat /home/flash/sc-draco.url) --header "Content-Type:application/json"


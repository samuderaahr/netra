#!/bin/sh

# Print timestamp

TSTAMP=$(date +%s)
echo "TSTP: $TSTAMP"
echo "ts,$TSTAMP" >> /var/log/telemetry

# Read Jetson Nano Temps

TZ00=$(cat /sys/devices/virtual/thermal/thermal_zone0/temp | cut -c1-3 | sed 's/./&./2')
CPUT=$(cat /sys/devices/virtual/thermal/thermal_zone1/temp | cut -c1-3 | sed 's/./&./2')
GPUT=$(cat /sys/devices/virtual/thermal/thermal_zone2/temp | cut -c1-3 | sed 's/./&./2')
PLLT=$(cat /sys/devices/virtual/thermal/thermal_zone3/temp | cut -c1-3 | sed 's/./&./2')
AVGT=$(cat /sys/devices/virtual/thermal/thermal_zone5/temp | cut -c1-3 | sed 's/./&./2')

echo "TZ00: $TZ00"
echo "CPUT: $CPUT"
echo "GPUT: $GPUT"
echo "PLLT: $PLLT"
echo "AVGT: $AVGT"

# Read Jetson Nano Power Draw

TOTP=$(cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input)
GPUP=$(cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power1_input)
CPUP=$(cat /sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power2_input)

echo "CPUP: $CPUP"
echo "GPUP: $GPUP"
echo "TOTP: $TOTP"

# Read Coral USB Temp (PCIe only ):

# Read HDD Temps

if [ -e /dev/sda ];
then
	HDDTEMP1=$(sudo smartctl -d sat -a /dev/sda | grep Celsius | awk {'print $10'})
	echo "HDD1: $HDDTEMP1.0"
else
	echo "HDD1 does not exist !!!"
	#TODO: log to critical error
fi

if [ -e /dev/sdb ];
then
	HDDTEMP2=$(sudo smartctl -d sat -a /dev/sdb | grep Celsius | awk {'print $10'})
	echo "HDD2: $HDDTEMP2.0"
else
	echo "HDD2 does not exist"
fi

# Get Free Memory and Swap
FREERAM=$(free --mega | grep Mem | awk {'print($7)'})
FREESWAP=$(free --mega | grep Swap | awk {'print($3)'})

echo "FREE: $FREERAM"
echo "SWAP: $FREESWAP"

# Get CPU Usage
CPUU=$(mpstat 1 1 | grep Average | awk {'print($12)'} | tr , .)
CPUU=$(python -c "print(100.00 - $CPUU)")
echo "CPUU: $CPUU%"

# Get GPU Usage
GPUU=$(cat /sys/devices/platform/host1x/57000000.gpu/load)
echo "GPUU: $GPUU%"

# Get SD card free space
DFSD=$(df -h | grep mmcblk | awk {'print($4)'} | tr , .)
DFAR=$(df -h | grep archive | awk {'print($4)'} | tr , .)

echo "uSDF: $DFSD Bytes"
echo "ArcF: $DFAR Bytes"

# Write all data to log CSV
# ts,tz00,cpuT,gpuT,pllT,avgT,cpuP,gpuP,totP,hdd1,hdd2,free,swap,cpuU,gpuU,dfSD,dfAR

sudo echo "$TSTAMP,$TZ00,$CPUT,$GPUT,$PLLT,$AVGT,$CPUP,$GPUP,$TOTP,$HDDTEMP1.0,$HDDTEMP2.0,$FREERAM,$FREESWAP,$CPUU%,$GPUU%,$DFSD,$DFAR" >> /var/log/telemetry.csv

# Convert last entry of CSV to JSON

sudo echo $(cat /var/log/telemetry.csv | head -n1) > /var/log/buffer.csv
sudo echo $(cat /var/log/telemetry.csv | tail -n1) >> /var/log/buffer.csv
sudo csvjson /var/log/buffer.csv > /var/log/payload.json

curl -v -X POST -d @/var/log/payload.json --url http://sc-draco.flashcoffeetech.com:8080/api/v1/ruXHnIlTv4tTN30jycl1/telemetry --header "Content-Type:application/json"
#TODO: Use SSL
#TODO: Use connection timeout
#TODO: Handle transport errors and retries

# Line separator
#echo 
#echo "\n---------------------\n" >> /var/log/telemetry

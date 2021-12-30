#!/bin/sh
ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 $1
ffprobe -i $1 -show_entries format=duration -v quiet -of csv="p=0"

#!/bin/bash

shopt -s "dotglob"
shopt -s "nullglob"
mkdir $1
for i in /media/pibrain/extern/reagan/*
do
  mkdir $1/$(basename ${i%%})
  ffmpeg -i "./${i}" -f wav -ab 192000 -ar 16000 -vn $1/$(basename ${i%%})/output.wav
  ffmpeg -i $1/$(basename ${i%%})/output.wav -f wav -map_channel 0.0.0 $1/$(basename ${i%%})/left.wav -map_channel 0.0.1 $1/$(basename ${i%%})/right.wav
  ffmpeg-normalize $1/$(basename ${i%%})/left.wav
  ffmpeg-normalize $1/$(basename ${i%%})/right.wav
done

#!/bin/bash

shopt -s "dotglob"
shopt -s "nullglob"
mkdir 1
for i in ./reagan/*
do
  mkdir $1/$(basename ${i%%})
  ffmpeg -i "./${i}" -qscale:v 2 -vf scale=720x406,setsar=1:1 $1/$(basename ${i%%})/output-%04d.jpg
done

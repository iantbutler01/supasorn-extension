#!/bin/bash

shopt -s "dotglob"
shopt -s "nullglob"
mkdir /Volumes/extern/out
for i in ./reagan/*
do
  mkdir /Volumes/extern/out/$(basename ${i%%})
  ffmpeg -i "./${i}" -qscale:v 2 -vf scale=720x406,setsar=1:1 /Volumes/extern/out/$(basename ${i%%})/output-%04d.jpg
done

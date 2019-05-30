#!/bin/bash

if [ $# -ne 3 ]
then
    echo "Usage: $0 audio_file ip port"
    exit -1
fi

file=$1

uuid=$(uuidgen)
len=$(stat -c "%s" $file)

asr_result=$(curl -s -X POST --data-binary @$file "http://$2:$3/recognize.cgi?vid=${uuid}&uid=experience&token=7f1f8f4647a24051a4b381a7a6108631&appkey=fc03efeeeb7e11e78c3f9a214cf093ae&bits=16&channel=1&fmt=1&rate=16000&seq=0&len=${len}&end=1" | grep text | cut -d '"' -f 4)
filename=$(basename $file)
echo "${filename%.*}:${asr_result}"

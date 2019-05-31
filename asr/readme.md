As described in our paper, we directly use an ASR api  to convert the audio data to the text and then select those incorrectly detected results by comparing the golden text data as our data. A simple script is provided  in this folder, and the users can pass their own ASR api address (ip+port) to do the audio-to-text  convertion:

## Usage:

```
 bash audio_file.sh ip port
 
 ip:   the ip address of the asr service, i.e., 127.0.0.1
 port: the port of the asr service, i.e., 8080
```


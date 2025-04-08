# tts-fish-speech
tts-fish-speech for HeyGem tts, docker free, support Linux, offline.  
一个离线版，与 HeyGem 一起使用的 tts 模型，免 docker, 支持 Linux。  

## note
本项目来自于 [fish-speech](https://github.com/fishaudio/fish-speech), 适配了 python 3.8，以支持与 HeyGem 同环境使用。
## 环境
参考 [requirements.txt](requirements.txt)

## 安装
```bash
git clone https://github.com/Holasyb918/tts-fish-speech
cd tts-fish-speech
# 下载模型
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir checkpoints/fish-speech-1.5/
```

## 运行示例
把你需要生成的文本放在 [asserts/text.txt](asserts/text.txt) 中，把要克隆的音色放在 [asserts/audio.wav](asserts/audio.wav) 中，然后运行以下命令：
```bash
bash run.sh asserts/audio.wav asserts/text.txt
#             音色 wav           TTS 文本 
```

## star us
If you find this project useful, please star us.  
如果这个项目对您有帮助，请给我们一颗 star。

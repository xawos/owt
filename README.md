<div  align="center">
<br>
<a  href="">
<img  src="res/github/ollama-telegram-readme.png"  width="200"  height="200">
</a>
<h1>🦙🗣️ Ollama Telegram Bot</h1>
<p>
<b>Chat **and speak** to your LLM, using Telegram bot!</b><br>
<b>Feel free to contribute!</b><br>
</p>
</div>

Shamelessly forked from [Ollama-telegram](https://github.com/ruecat/ollama-telegram).

I only added the voice part and couple of things such as preregistering users once they appear in the `.env ` file at bot start.

## Prerequisites
- [Telegram-Bot Token](https://core.telegram.org/bots#6-botfather)
  
## Installation (Non-Docker)
+ Clone Repository
```
git clone https://github.com/xawos/owt
```
+ Install requirements from requirements.txt
```
pip install -r requirements.txt
```
+ Enter all values in .env.example as shown in the [original repo](https://github.com/ruecat/ollama-telegram)  for starters.

NB: My fork has 2 additional flags, `WSPRMODEL` and `USE_CUDA`, respectively set to `base` and `True`.

Both those options are meant for [Whisper](https://github.com/openai/whisper), list of models [here](https://github.com/openai/whisper?tab=readme-ov-file#available-models-and-languages). 
  
+ Rename `.env.example` -> `.env` after setting the above options
  
+ Launch bot and wait until you see a line starting with `INFO:aiogram.dispatcher:Run polling for bot`
  
```
python3 run.py
```
If everything went well good job! You can now send voice messages and images to your Telegram bot!

It also works on Raspberry Pi, with [Phi3](https://ollama.com/library/phi3) (without vision) it replies to a voice prompt in ~30s!

  
  
  

## Credits

+ [Ollama-telegram](https://github.com/ruecat/ollama-telegram) (original bot by [ruecat](https://github.com/ruecat/))
+ [Ollama](https://github.com/jmorganca/ollama)
+ [Whisper](https://github.com/openai/whisper)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)  

## Libraries used

+ [Aiogram 3.x](https://github.com/aiogram/aiogram)
+ [OpenAI-Whisper](https://pypi.org/project/openai-whisper/)
+ [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)
+ [and more](https://github.com/xawos/owt/blob/main/requirements.txt)

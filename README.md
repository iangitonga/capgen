# capgen
Capgen is an automatic captions generator for videos and audio. It employs [Whisper](https://openai.com/blog/whisper/) neural network,
offered by OpenAI to generate accurate timestamped captions for your videos and audio in text, srt and vtt file formats.

## Colab Example
To observe capgen in action or play with it, open the colab notebook [here](https://colab.research.google.com/drive/1O1FIQVogzoPlXmf4xZ-mn6q48M5jzXeG?usp=sharing)

## Dependencies
- [ffmpeg](https://ffmpeg.org/download.html) for decoding all video formats.

## Installing capgen
1. Download and install [ffmpeg](https://ffmpeg.org/download.html).
2. Clone this repository by running `git clone https://github.com/iangitonga/capgen.git`
3. Run `pip install -r requirements.txt` to install Python dependencies.

## Usage
- Run `python capgen.py <path-to-video>` to generate a captions files.
- Run `python capgen.py <path-to-video> --task translate` to translate captions to English.	
- Run `python capgen.py --help` to see all the available options.
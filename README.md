# capgen
Capgen is an automatic captions generator for videos. It employs Whisper neural network,
offered by OpenAI to generate accurate timestamped captions for your videos in srt file format.
Whisper is introduced here, https://openai.com/blog/whisper/. The code for Capgen is a from-scratch
re-implementation of the code released by OpenAI. Support for other output formats will be added in the future👀.

## Colab Example
To observe capgen in action or play with it, open the colab notebook [here](https://colab.research.google.com/drive/1O1FIQVogzoPlXmf4xZ-mn6q48M5jzXeG?usp=sharing)

## Dependencies
- [ffmpeg](https://ffmpeg.org/download.html) for handling all types of videos.

## Installing capgen
1. Download and install [ffmpeg](https://ffmpeg.org/download.html).
2. Clone this repository by running `git clone https://github.com/iangitonga/capgen.git`.
3. Run `pip install -r requirements.txt` to install Python dependencies.

## Usage
- Run `python capgen.py <path-to-video>` to generate a captions file.
- Run `python capgen.py --help` to see all the available options.
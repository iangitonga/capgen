# capgen
Capgen is an automatic captions generator for videos. It employs Whisper neural network,
offered by OpenAI to generate accurate timestamped captions for your videos in srt file format.
Whisper is introduced here, https://openai.com/blog/whisper/. The code for capgen is a from-scratch
re-implementation of the code released by OpenAI. The current implementation utilizes `tiny.en` model
which is the smallest english-only model OpenAI trained. Support for all the models and translation
and output formarts will be added in the future👀.

## Dependencies
- ffmpeg cmd program for extracting audio from video available at https://ffmpeg.org/download.html

## Installing
1. Download the source from Github.
2. `cd` to capgen\capgen directory.
3. Download ffmpeg cmd.
3. Run `pip install -r requirements.txt` to install dependencies.

## Usage
Run `python capgen.py <path-to-video>`.
An srt file is generated in the same directory.
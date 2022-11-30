# Capgen
Capgen is an automatic captions generator for videos. It employs Whisper neural network (https://openai.com/blog/whisper/)
offered by OpenAI to generate accurate timestamped captions for your videos in srt file format.

# Dependencies
- ffmpeg cmd program available at https://ffmpeg.org/download.html
- ffmpeg-python
- PyTorch
- Numpy
- transformers library from HuggingFace
- tqdm

# Installing
1. Download the source from Github.
2. cd to capgen directory.
3. Download ffmpeg.
3. Run `pip -r requirements.txt`

# Usage
Run `python capgen.py <path-to-video>`
An srt file is generated in the same directory.
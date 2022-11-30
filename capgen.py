import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from loader import load_model
from audio import load_audio_from_video, get_audio_mel_spectrogram, get_spectrogram_chunks
from text import GreedyDecoder, save_to_srt
from tokenizer import get_tokenizer


def transcribe(filepath):
    # TODO:  allow for GPU and model selection.
    model = load_model('tiny.en', device='cpu')
    audio = load_audio_from_video(filepath)
    mel = get_audio_mel_spectrogram(audio)
    mel_chunks = get_spectrogram_chunks(mel)

    # TODO:  allow for multilingual tokenizer.
    tokenizer = get_tokenizer(multilingual=False, task='transcribe', language='en')
    # TODO:  allow for decoder specification.
    decoder = GreedyDecoder(model, tokenizer, max_sample_len=224)
    tokens = decoder.decode(mel_chunks)
    abspath = os.path.abspath(filepath)
    basepath, filename = os.path.split(abspath)
    srt_filename = filename.split('.')[0] + '.srt'
    srt_filepath = os.path.join(basepath, srt_filename)
    save_to_srt(tokens, tokenizer, srt_filepath)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('Error: filepath was not provided.')
        exit(-1)
    # TODO: Allow for multiple paths.
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: provided filepath '{filepath}' does not exist.")
        exit(-1)
    if not os.path.isfile(filepath):
        print(f"Error: provided filepath '{filepath}' is not a file.")
        exit(-1)
    transcribe(filepath)

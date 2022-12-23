import argparse
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages.
import torch
import tqdm
import numpy as np

from audio import load_audio_from_video, get_audio_mel_spectrogram, pad_or_trim_spectrogram
from decoders import detect_language, save_to_srt, GreedyDecoder, BeamSearchDecoder, SamplingDecoder, AVAILABLE_DECODERS
from loader import load_model, AVAILABLE_MODELS
from tokenizer import get_tokenizer, LANGUAGES

# Maximum compression ratio a segment transcription text should have. Greater
# compression ratio than this means the text is too repetitive.
COMPRESSION_RATIO_THRESHOLD = 2.4


def transcribe(filepath: str, model_name: str, task: str, language: str, decoder_type: str, n_beam: str, temperature: float):
    # TODO:  allow for model selection.
    model = load_model(model_name)
    audio = load_audio_from_video(filepath).to(model.device)
    mel_spectrogram = get_audio_mel_spectrogram(audio)

    # Tokenizer loading and language detection.
    if model_name.endswith('en'):
        language = 'en'
        tokenizer = get_tokenizer(multilingual=False, task='transcribe', language=language)
    elif language:
        tokenizer = get_tokenizer(multilingual=True, task=task, language=language)
    else:
        print('Performing language detection ...', end='')
        language = detect_language(mel_spectrogram, model)
        print(f'language detected: {LANGUAGES[language]}')
        tokenizer = get_tokenizer(multilingual=True, task=task, language=language)

    # Decoder selection
    if decoder_type == 'greedy':
        decoder = GreedyDecoder(model, tokenizer)
    elif decoder_type == 'beamsearch':
        decoder = BeamSearchDecoder(model, tokenizer, n_beam)
    elif decoder_type == 'sampling':
        decoder = SamplingDecoder(model, tokenizer, temperature)

    # Transcription process.
    print('Transcribing ...')
    current_segment_pos = 0  # Starting position for the segment we are about to transcribe.
    audio_transcript = []
    n_segments = (mel_spectrogram.shape[1] // 3000) + 1
    with tqdm.tqdm(total=n_segments, ncols=80) as progbar:
        while True:
            if (mel_spectrogram.shape[1] - current_segment_pos) < 2000:
                break
            audio_segment = pad_or_trim_spectrogram(mel_spectrogram[:,current_segment_pos:])
            result = decoder.decode_segment(audio_segment)
            # If the transcription results are too repetitive we use sampling with various temperatures
            # as fallback.
            if result.compression_ratio > COMPRESSION_RATIO_THRESHOLD:
                for temperature in (0.2, 0.4, 0.6, 0.8, 1.0):
                    result = SamplingDecoder(model, tokenizer, temperature).decode_segment(audio_segment)
                    if result.compression_ratio <= COMPRESSION_RATIO_THRESHOLD:
                        break
            audio_transcript.append(result)
            current_segment_pos += int(result.final_timestamp * 100)

            progbar.update(1)

    abspath = os.path.abspath(filepath)
    basepath, filename = os.path.split(abspath)
    srt_filename = filename.split('.')[0] + '.srt'
    srt_filepath = os.path.join(basepath, srt_filename)
    save_to_srt(audio_transcript, tokenizer, srt_filepath)


def _validate_filepath(filepath):
    if not os.path.exists(filepath):
        print(f"Error: provided filepath '{filepath}' does not exist.")
        exit(-1)
    if not os.path.isfile(filepath):
        print(f"Error: provided filepath '{filepath}' is not a file.")
        exit(-1)
    return filepath


def _validate_nbeam(n_beam):
    if n_beam < 1:
        print(f"Error: number of beams cannot be less than one. You provided '{n_beam}'.")
        exit(-1)
    return n_beam

def _validate_temperature(temperature):
    if temperature <= 0:
        print(f"Error: temperature must be greater than zero. You provided '{temperature}'.")
        exit(-1)
    return temperature

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='path to the video supposed to be transcribed')
    parser.add_argument(
        '--model',
        help="type of model to use. Default is medium model.",
        default='medium',
        choices=AVAILABLE_MODELS,
    )
    parser.add_argument(
        '--task',
        help="task to perform. 'transcribe' produces captions in the language spoken in video while 'translate' translates to English. Default is transcribe.",
        default='transcribe',
        choices=('transcribe', 'translate'),
    )
    parser.add_argument(
        '--language',
        help="language spoken in the video. If not provided, it is automatically detected.",
        default='',
        choices=sorted(LANGUAGES.keys()),
    )
    parser.add_argument(
        '--decoder',
        help="type of decoder to use to perform transcription. Default is beamsearch which produces better transcriptions.",
        default='beamsearch',
        choices=AVAILABLE_DECODERS,
    )
    parser.add_argument(
        '--nbeam',
        help="number of beams to use when using beamsearch decoder. Default is 5.",
        type=int,
        default=5,
    )
    parser.add_argument(
        '--temperature',
        help="temperature to use when using sampling decoder. Default is 1.",
        type=float,
        default=1.,
    )

    args = parser.parse_args()
    filepath = _validate_filepath(args.filepath)
    model_name = args.model
    task = args.task
    language = args.language
    decoder = args.decoder
    n_beam = _validate_nbeam(args.nbeam)
    temperature = _validate_temperature(args.temperature)

    if decoder == 'greedy':
        print(f'Performing transcription on {filepath} using <greedy> decoder.')
    elif decoder == 'sampling':
        print(f'Performing transcription on {filepath} using <sampling> decoder with temperature: {temperature}')
    else:
        print(f'Performing transcription on {filepath} using <{decoder}> decoder with {n_beam} beams.')

    if model_name.endswith('en') and language and language != 'en':
        print(f"Warning: the model provided is English-only and cannot transcribe provided language '{language}'.")
        exit(-1)

    # Perform transcription
    transcribe(filepath, model_name, task, language, decoder, n_beam, temperature)


cli()
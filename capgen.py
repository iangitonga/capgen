import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages.

from loader import load_model, AVAILABLE_MODELS
from audio import load_audio_from_video, get_audio_mel_spectrogram
from decoders import BeamSearchDecoder, save_to_srt, AVAILABLE_DECODERS

from tokenizer import get_tokenizer



def transcribe(filepath, model_name, decoder_type, n_beam):
    # TODO:  allow for model selection.
    print('Loading model and audio ...')
    model = load_model(model_name)
    audio = load_audio_from_video(filepath)
    mel_spectrogram = get_audio_mel_spectrogram(audio)

    # TODO:  allow for multilingual tokenizer.
    tokenizer = get_tokenizer(multilingual=False, task='transcribe', language='en')
    decoder = BeamSearchDecoder(model, tokenizer, max_sample_len=224)
    if decoder_type == 'greedy':
        tokens = decoder.decode(mel_spectrogram, n_beam=1)
    else:
        tokens = decoder.decode(mel_spectrogram, n_beam=n_beam)
    abspath = os.path.abspath(filepath)
    basepath, filename = os.path.split(abspath)
    srt_filename = filename.split('.')[0] + '.srt'
    srt_filepath = os.path.join(basepath, srt_filename)
    save_to_srt(tokens, tokenizer, srt_filepath)



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

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='path to the video supposed to be transcribed')
    parser.add_argument(
        '--model',
        help="type of model to use. Currently only english models are supported. Default is medium model",
        default='medium.en',
        choices=AVAILABLE_MODELS,
    )
    parser.add_argument(
        '--decoder',
        help="type of decoder to use to perform transcription. Default is beamsearch which produces better transcriptions.",
        default='beamsearch',
        choices=AVAILABLE_DECODERS,
    )
    parser.add_argument(
        '--nbeam',
        help="number of beams to use when performing beamsearch. Default is 5. Ignored when decoder is not beamsearch. Large number of beams can slow transcription process.",
        type=int,
        default=5,
    )

    args = parser.parse_args()
    filepath = _validate_filepath(args.filepath)
    decoder = args.decoder
    n_beam = _validate_nbeam(args.nbeam)
    model_name = args.model

    if decoder == 'greedy':
        print(f'Performing transcription on {filepath} using <greedy> decoder.')
    else:
        print(f'Performing transcription on {filepath} using <{decoder}> decoder with {n_beam} beams.')

    # Perform transcription
    transcribe(filepath, model_name, decoder, n_beam)


cli()
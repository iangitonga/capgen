import argparse
import os

from capgen.transcribe import AVAILABLE_MODELS, AVAILABLE_DECODERS, LANGUAGES, TranscriptionOptions, transcribe


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

def _validate_language_and_model_compat(language, model_name):
    if model_name.endswith('en') and language and language != 'en':
        print(f"Warning: the model provided is English-only and cannot transcribe provided language '{language}'.")
        exit(-1)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="path to the video supposed to be transcribed")
    parser.add_argument(
        "--model",
        help="type of model to use. Default is medium model.",
        default="medium",
        choices=AVAILABLE_MODELS,
    )
    parser.add_argument(
        "--task",
        help="task to perform. 'transcribe' produces captions in the language spoken in video while 'translate' translates to English. Default is transcribe.",
        default="transcribe",
        choices=("transcribe", "translate"),
    )
    parser.add_argument(
        "--language",
        help="language spoken in the video. If not provided, it is automatically detected.",
        default="",
        choices=sorted(LANGUAGES.keys()),
    )
    parser.add_argument(
        "--decoder",
        help="type of decoder to use to perform transcription. Default is beamsearch which produces better transcriptions.",
        default="beamsearch",
        choices=AVAILABLE_DECODERS,
    )
    parser.add_argument(
        "--nbeam",
        help="number of beams to use when using beamsearch decoder. Default is 5.",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--temperature",
        help="temperature to use when using sampling decoder. Default is 1.",
        type=float,
        default=1.,
    )

    args = parser.parse_args()

    transcription_options = TranscriptionOptions(
        video_filepath = _validate_filepath(args.filepath),
        model_name = args.model,
        task = args.task,
        language = args.language,
        decoder = args.decoder,
        n_beam = _validate_nbeam(args.nbeam),
        temperature = _validate_temperature(args.temperature),
    )

    _validate_language_and_model_compat(args.language, args.model)

    # Perform transcription
    transcribe(transcription_options)


cli()
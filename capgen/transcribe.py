import os
from dataclasses import dataclass

import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow messages.

from .audio import load_audio_from_video, get_audio_mel_spectrogram, pad_or_trim_spectrogram
from .decoders import detect_language, GreedyDecoder, BeamSearchDecoder, SamplingDecoder, AVAILABLE_DECODERS
from .loader import load_model, AVAILABLE_MODELS
from .tokenizer import get_tokenizer, LANGUAGES
from .utils import save_to_srt, save_to_vtt, save_to_txt


# Maximum compression ratio a segment transcription text should have. Greater
# compression ratio than this means the text is too repetitive.
COMPRESSION_RATIO_THRESHOLD = 2.4
# TODO: Add logprob and no-speech threshold.


@dataclass
class TranscriptionOptions:
	video_filepath: str = None
	model_name: str = None
	task: str = None
	language: str = None
	decoder: str = None
	n_beam: str = None
	temperature: float = None


def _get_output_filename(video_filepath, extension):
    abspath = os.path.abspath(video_filepath)
    basepath, filename = os.path.split(abspath)
    filename = filename.split('.')[0] + '.' + extension
    return os.path.join(basepath, filename)


def transcribe(options: TranscriptionOptions):
    # TODO:  allow for model selection.
    model = load_model(options.model_name)
    audio = load_audio_from_video(options.video_filepath).to(model.device)
    mel_spectrogram = get_audio_mel_spectrogram(audio)

    # Tokenizer loading and language detection.
    if options.model_name.endswith("en"):
        tokenizer = get_tokenizer(multilingual=False, task="transcribe", language="en")
    elif options.language:
        tokenizer = get_tokenizer(multilingual=True, task=options.task, language=options.language)
    else:
        language = detect_language(mel_spectrogram, model)
        print(f"Detected language: {LANGUAGES[language]}")
        tokenizer = get_tokenizer(multilingual=True, task=options.task, language=language)

    # Decoder selection
    if options.decoder == 'greedy':
        decoder = GreedyDecoder(model, tokenizer)
    elif options.decoder == 'beamsearch':
        decoder = BeamSearchDecoder(model, tokenizer, options.n_beam)
    elif options.decoder == 'sampling':
        decoder = SamplingDecoder(model, tokenizer, options.temperature)

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

    # Save the transcripts.
    srt_fname = _get_output_filename(options.video_filepath, 'srt')
    save_to_srt(audio_transcript, tokenizer, srt_fname)
    vtt_fname = _get_output_filename(options.video_filepath, 'vtt')
    save_to_vtt(audio_transcript, tokenizer, vtt_fname)
    txt_fname = _get_output_filename(options.video_filepath, 'txt')
    save_to_txt(audio_transcript, tokenizer, txt_fname)
    print('\nCaptions successfully generated.')

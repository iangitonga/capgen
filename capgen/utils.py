from typing import Tuple

from .decoders import AudioSegmentTranscript
from .tokenizer import Tokenizer


def format_timestamp(timestamp: float, separator: str) -> str:
    milliseconds = round(timestamp * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{milliseconds:03d}"


def get_timestamp(token, tokenizer):
    return float(tokenizer.decode_with_timestamps([token]).split('|')[1])


def save_to_srt(segment_transcripts: Tuple[AudioSegmentTranscript], tokenizer: Tokenizer, fname: str) -> None:
    """Decodes the given audio transcription tokens and saves them in srt format.
    
    Args:
        segment_transcripts: A list of segment transcripts.
        tokenizer: A tokenizer to decode the tokens.
        fname: The name of the file to save the tokens.
    """
    with open(fname, 'w') as f:
        ts_tokens_ctr = 1
        segment_offset = 0
        for i in range(len(segment_transcripts)):
            segment_transcript = segment_transcripts[i]
            for ts_tokens in segment_transcript.segmented_tokens:
                index = str(ts_tokens_ctr)
                start_ts = get_timestamp(ts_tokens[0], tokenizer)
                start_time = format_timestamp(start_ts + segment_offset, separator=',')
                end_ts = get_timestamp(ts_tokens[-1], tokenizer)
                end_time = format_timestamp(end_ts + segment_offset, separator=',')
                timestamp = f'{start_time} --> {end_time}'
                text = tokenizer.decode(ts_tokens[1:-1]).lstrip()
                srt_segment = f'{index}\n{timestamp}\n{text}\n\n'
                f.write(srt_segment)
                ts_tokens_ctr += 1
            segment_offset += end_ts


def save_to_vtt(segment_transcripts: Tuple[AudioSegmentTranscript], tokenizer: Tokenizer, fname: str) -> None:
    """Decodes the given audio transcription tokens and saves them in vtt format.
    
    Args:
        segment_transcripts: A list of segment transcripts.
        tokenizer: A tokenizer to decode the tokens.
        fname: The name of the file to save the captions.
    """
    with open(fname, 'w') as f:
        f.write('WEBVTT\n\n')
        segment_offset = 0
        for i in range(len(segment_transcripts)):
            segment_transcript = segment_transcripts[i]
            for ts_tokens in segment_transcript.segmented_tokens:
                start_ts = get_timestamp(ts_tokens[0], tokenizer)
                start_time = format_timestamp(start_ts + segment_offset, separator='.')
                end_ts = get_timestamp(ts_tokens[-1], tokenizer)
                end_time = format_timestamp(end_ts + segment_offset, separator='.')
                timestamp = f'{start_time} --> {end_time}'
                text = tokenizer.decode(ts_tokens[1:-1]).lstrip()
                srt_segment = f'{timestamp}\n{text}\n\n'
                f.write(srt_segment)
            segment_offset += end_ts


def save_to_txt(segment_transcripts: Tuple[AudioSegmentTranscript], tokenizer: Tokenizer, fname: str) -> None:
    """Decodes the given audio transcription tokens and saves them in text format without timestamps."""
    with open(fname, 'w') as f:
        for segment_transcript in segment_transcripts:
            f.write(tokenizer.decode(segment_transcript.tokens) + '\n\n')

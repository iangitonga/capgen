"""Handles Audio processing tasks."""

import os

import ffmpeg
import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor


# CONSTANTS
SAMPLE_RATE = 16000
# Window size: the number of samples on which to perform a discrete fourier transform
# Given the sampling rate of 16000Hz, this corresponds to 0.025s or 25ms of audio.
N_FFT = 400
# Hop length is how many data points to skip after performing dft on a given window size.
# Essentially, it is the distance between neighboring sliding windows.
HOP_LENGTH = 160
# Number of mel bin frequency bands.
N_MELS = 80
CHUNK_LENGTH = 30 # in secs
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
# Number of frames in a mel spectrogram output expected by the model(3000).
N_FRAMES = N_SAMPLES // HOP_LENGTH
# Number of frames in the encoder input representing one second of audio.
N_FRAMES_PER_SECOND = N_FRAMES / CHUNK_LENGTH


def load_audio_from_video(filepath: str) -> Tensor:
    """Extracts audio from a video in the given path.

    Args:
        filepath: Path to the video from which the audio shall be extracted.
    
    Returns:
        The audio in a tensor of shape (n_audio_samples,) if the extraction is successful. Otherwise an
        error message is shown and the program exits.
    """
    try:
        stream = ffmpeg.input(filepath)
        audio_stream = stream['a']  # Pluck the audio stream.
        # `s16le` format is PCM signed 16-bit little-endian which represents a raw audio format
        # that can be processed the model. The `-` represents the output filename.
        output_stream = ffmpeg.output(audio_stream, '-', format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
        audio_buffer, _ = ffmpeg.run(output_stream, cmd=['ffmpeg', '-nostdin'], capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    audio = np.frombuffer(audio_buffer, np.int16)
    audio = audio / audio.max()  # Normalize.
    audio = audio.astype(np.float32)
    return torch.from_numpy(audio)


def mel_filters(device) -> torch.Tensor:
    """Load the mel filterbank matrix for projecting STFT into a Mel spectrogram.

    The filterbank was computed using Librosa library.

    Returns:
        A filterbank tensor of shape (80, 201). 80 is the number of dimensions to project
        STFT output and 201 = (N_FFT // 2) + 1 is the number of frequencies computed by STFT. 
    """
    path = os.path.join(os.path.dirname(__file__), "assets", 'mel_filters.npz')
    with np.load(path) as f:
        return torch.from_numpy(f['mel_80']).to(device)


def get_audio_mel_spectrogram(audio: Tensor) -> Tensor:
    """Computes the mel-spectrogram of the audio in the given path.

    Args:
        audio: a tensor of the audio to be processed.
    Returns:
        a tensor of shape (N_MELS, N_FRAMES) representing the 2D spectogram.
    """
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    # The code below ditches the last column, takes the abs value(magnitudes) of freqs and squares them.
    # The last column belongs to the frequencies of the last window/frame.
    magnitudes = stft[:, :-1].abs() ** 2
    # Returns a filter matrix which can projects frequency bins to mel bins.
    filters = mel_filters(audio.device)
    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def pad_or_trim_spectrogram(spectrogram: Tensor, n_frames: int = N_FRAMES, *, axis: int = -1) -> Tensor:
    """Pads or trims the given spectogram to n_frames, as expected by the encoder.

    Args:
        spectrogram: A tensor of shape (*, N) where N is the number of frames the spectrogram contains.
        n_frames: The number of frames in the output. Default is 3000.
        axis: The axis where the frames lie. Default is -1, last axis.

    Raises:
        TypeError: If the input spectogram is not a Tensor.
        
    Returns:
        A tensor of shape (*, n_frames) representing trimmed or padded spectrogram.
    """
    if torch.is_tensor(spectrogram):
        audio_frames = spectrogram.shape[axis]
        if audio_frames > n_frames:
            # Selects the first n expected frames.
            spectrogram = spectrogram.index_select(dim=axis, index=torch.arange(n_frames, device=spectrogram.device))

        if audio_frames < n_frames:
            # Appends <diff> cols/frames full of zeros.
            pad_widths = [(0, 0)] * spectrogram.ndim
            diff = n_frames - audio_frames
            pad_widths[axis] = (0, diff)
            spectrogram = F.pad(spectrogram, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        raise TypeError(f'Expected a torch tensor, found: {type(spectrogram)}')
    spectrogram = spectrogram.unsqueeze(0)  # Add a batch dimension
    return spectrogram

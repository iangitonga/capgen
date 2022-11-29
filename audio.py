"""Handles Audio processing tasks."""

import os

import ffmpeg
import torch
import torch.nn.functional as F
import numpy as np


# Typing stuff.
Tensor = torch.Tensor

# CONSTANTS
SAMPLE_RATE = 16000
# Window size: the number of samples on which to perform a discrete fourier transform
# Given the sampling rate of 16000Hz, this corresponds to 0.025s or 25ms of audio.
N_FFT = 400
# Hop length is how many data points to skip after performing dft on a given window size.
# Hop length of one means after performing a dft we slide the window by one step. Essentially
# it is the distance between neighboring sliding windows. Hop length greater than or equal to
# N_FFT would mean no overlap. The less the hop_length the more the overlap.
HOP_LENGTH = 160
# Number of mel bin frequency bands.
N_MELS = 80
CHUNK_LENGTH = 30 # in secs
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
# Number of frames in a mel spectrogram output expected by the model(3000).
N_FRAMES = N_SAMPLES // HOP_LENGTH


def load_audio_from_video(filepath):
    stream = ffmpeg.input(filepath)
    audio_stream = stream['a']  # Pluck the audio stream.
    # `s16le` format is PCM signed 16-bit little-endian which represents a raw audio format
    # that can be processed the model. The `-` represents the output filename.
    output_stream = ffmpeg.output(audio_stream, '-', format="s16le", acodec="pcm_s16le", ac=1, ar=SAMPLE_RATE)
    audio_buffer, _ = ffmpeg.run(output_stream, cmd=['ffmpeg', '-nostdin'], capture_stdout=True, capture_stderr=True)
    audio = np.frombuffer(audio_buffer, np.int16)
    audio = audio / audio.max()  # Normalize.
    audio = audio.astype(np.float32)
    return torch.from_numpy(audio)


def mel_filters() -> torch.Tensor:
    """Load the mel filterbank matrix for projecting STFT into a Mel spectrogram. """
    with np.load('mel_filters.npz') as f:
        return torch.from_numpy(f['mel_80'])


def get_audio_mel_spectrogram(audio: Tensor) -> Tensor:
    """Computes the mel-spectrogram of the audio in the given path.

    Args:
        audio: a tensor of the audio to be processed.
    Returns:
        a tensor of shape (N_MELS, N_FRAMES) representing the 2D spectogram.
    """
    window = torch.hann_window(N_FFT)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    # The code below ditches the last column, takes the abs value(magnitudes) of freqs and squares them.
    # The last column belongs to the frequencies of the last window/frame.
    magnitudes = stft[:, :-1].abs() ** 2
    # Returns a filter matrix which can projects frequency bins to mel bins.
    filters = mel_filters()
    mel_spec = filters @ magnitudes
    # Remove values less than 1e-10 and convert to log 10
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    # Clip all the values less than {log_spec.max() - 8.0}
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def pad_or_trim_spectrogram(spectrogram: Tensor, n_frames: int = N_FRAMES, *, axis: int = -1) -> Tensor:
    """Pads or trims the given spectogram to n_frames, as expected by the encoder.

    Args:
        spectrogram: a tensor of shape (*, N) where N is the number of frames the spectrogram contains.
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
    return spectrogram


def get_spectrogram_chunks(spectrogram):
    """Splits the given spectrogram into chunks of spectrograms with the number
     of frames expected by the encoder
    
    Args:
        spectrogram: The entire spectrogram of input audio of shape (n_mels, F) where
         F is the number of total frames in the audio.
    Returns:
        A batched tensor of spectrograms of shape (n_chunks, n_mels, T) where T is the number
         of frames expected by the encoder.
    """
    n_mels, T = spectrogram.shape
    n_chunks = T // N_FRAMES if T % N_FRAMES == 0 else T // N_FRAMES + 1
    chunks = torch.empty((n_chunks, n_mels, N_FRAMES))
    for i in range(n_chunks):
        chunks[i] = pad_or_trim_spectrogram(spectrogram[:,i*N_FRAMES:])
    return chunks

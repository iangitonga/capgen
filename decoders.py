"""Handles Text processing tasks."""

import abc
import time
import zlib
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from audio import pad_or_trim_spectrogram, N_FRAMES, N_FRAMES_PER_SECOND
from model import Whisper
from tokenizer import get_tokenizer, Tokenizer

# CONSTANTS
AVAILABLE_DECODERS = ('greedy', 'beamsearch', 'sampling')
MAX_SAMPLE_LEN = 448 // 2  # Maximum length of tokens in a 30s audio segment. Equal to n_ctx//2


class ModelInference:
    def __init__(self, model: Whisper):
        self.model = model
        self.cache = {}
        self.hooks = []
        
    def install_hooks(self):
        cache, hooks = self.model.install_kv_hooks(self.cache)
        self.cache = cache
        self.hooks = hooks
        
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.cache = {}
    
    def embed_audio(self, mel: Tensor) -> Tensor:
        return self.model.embed_audio(mel)
    
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        return self.model.logits(tokens, audio_features, self.cache)


@dataclass
class AudioSegmentTranscript:
    tokens: List[int]
    segmented_tokens: List[List[int]]
    compression_ratio: float
    final_timestamp: float


class Decoder(abc.ABC):
    def __init__(self, model: Whisper, tokenizer: Tokenizer, max_sample_len: int=MAX_SAMPLE_LEN):
        self.inference = ModelInference(model)
        self.tokenizer = tokenizer
        self.max_sample_len = max_sample_len
        self.suppress_tokens = self._get_suppress_tokens()
        self.sample_begin = len(self.tokenizer.sot_sequence)  # len of initial tokens.

    def decode_segment(self, spectrogram: Tensor) -> AudioSegmentTranscript:
        """Extracts transcriptions of the given audio segment from the model.

        Args:
            spectrogram: The spectrogram of the audio segment to be transcribed. It should be of
             shape (1, n_mels, 3000).

        Returns:
            Transcription results.
        """
        self.inference.install_hooks()
        result = self._decode_segment(spectrogram)
        self.inference.remove_hooks()
        return result

    @abc.abstractmethod
    def _decode_segment(self, spectrogram: Tensor) -> AudioSegmentTranscript:
        pass

    def suppress_logits(self, logits: Tensor, tokens: Tensor) -> None:
        self._suppress_tokens(logits, tokens)
        self._suppress_blank(logits, tokens)
        self._apply_timestamp_rules(logits, tokens)
    
    def _suppress_blank(self, logits: Tensor, tokens: Tensor) -> None:
        if tokens.shape[1] == 1:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
            
    def _suppress_tokens(self, logits: Tensor, tokens: Tensor) -> None:
        logits[:, self.suppress_tokens] = -np.inf
        
    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = []
        suppress_tokens.extend(self.tokenizer.non_speech_tokens)

        suppress_tokens.extend([self.tokenizer.sot, self.tokenizer.sot_prev, self.tokenizer.sot_lm])
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))
    
    def _apply_timestamp_rules(self, logits: Tensor, tokens: Tensor) -> None:
        # precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
        precision = 30 / 448
        max_initial_timestamp = 1.0  # the initial timestamp cannot be later than this
        max_initial_timestamp_index = None
        if max_initial_timestamp:
            max_initial_timestamp_index = round(max_initial_timestamp / precision)
            
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            logits[:, self.tokenizer.no_timestamps] = -np.inf

        # timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        for k in range(tokens.shape[0]):
            seq = [t for t in tokens[k, self.sample_begin :].tolist()]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin

            if last_was_timestamp:
                # timestamps have appeared in pairs.
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

        # apply the `max_initial_timestamp` option
        if tokens.shape[1] == self.sample_begin and max_initial_timestamp_index is not None:
            last_allowed = self.tokenizer.timestamp_begin + max_initial_timestamp_index
            logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf

    def get_final_timestamp(self, tokens: List[int]) -> float:
        for token in reversed(tokens):
            if token >= self.tokenizer.timestamp_begin:
                return float(self.tokenizer.decode_with_timestamps([token]).split('|')[1])
        raise ValueError(f'Timestamp token not found in {tokens}')

    def text_compression_ratio(self, text: bytes) -> float:
        return len(text) / len(zlib.compress(text))

    def segment_tokens(self, tokens: List[int]) -> List[List[int]]:
        """Given a list of tokens, it returns a list constaining lists where each inner list begins with
        a timestamp token, then text tokens and ends with timestamp token.
        """
        segment_tokens = []
        current_tokens = []
        for token in tokens:
            if token >= self.tokenizer.timestamp_begin and len(current_tokens) != 0:
                current_tokens.append(token)
                segment_tokens.append(current_tokens)
                current_tokens = []
            else:
                current_tokens.append(token)
        return segment_tokens

    def create_transcript(self, tokens: List[int]) -> AudioSegmentTranscript:
        transcript = AudioSegmentTranscript(
            tokens=tokens,
            segmented_tokens=self.segment_tokens(tokens),
            compression_ratio=self.text_compression_ratio(self.tokenizer.decode(tokens).encode('utf-8')),
            final_timestamp=self.get_final_timestamp(tokens),
        )
        return transcript



class GreedyDecoder(Decoder):
    """Implements greedy decoding method.

    Greedy decoding selects the token with the highest probability at each timestep. It has the
    advantage of being simple to implement and faster than beamsearch. It has the following
    drawbacks:
     - It does not necessarily produce the transcription with the highest probability.
     - It can easily get stuck in a repetition loop.
    """
    def _decode_segment(self, spectrogram):
        audio_features = self.inference.embed_audio(spectrogram)
        ctx_tokens = torch.tensor([[*self.tokenizer.sot_sequence]])
        out_tokens = []
        for i in range(self.max_sample_len):
            logits = self.inference.logits(ctx_tokens, audio_features)
            logits = logits[:,-1]
            self.suppress_logits(logits, ctx_tokens)
            probs = F.softmax(logits, dim=-1)
            pred_token = probs.argmax(dim=-1).item()
            if pred_token == self.tokenizer.eot:
                break
            ctx_tokens = torch.tensor([ctx_tokens[0].tolist() + [pred_token]])
            out_tokens.append(pred_token)
        result = self.create_transcript(out_tokens)
        return result


class BeamSearchDecoder(Decoder):
    """Implements standard beam search decoding method.

    The standard beam search generates tokens step-by-step by keeping a set of the B(beam width) highest-scoring
    beams generated so far at each step. Beam search is an approximate method to find the transcription with the
    highest probability assigned by the model. Though it is slower than greedy decoding, it always produces better
    transcriptions in practice and is less prone to repetition loops.
    """
    def __init__(self, model: Whisper, tokenizer: Tokenizer, n_beam: int):
        super().__init__(model, tokenizer)
        self.n_beam = n_beam

    def _decode_segment(self, spectrogram):
        audio_features = self.inference.embed_audio(spectrogram)
        # Copy the audio features to use for each beam. (n_beam, T, D)
        audio_features = audio_features.repeat_interleave(self.n_beam, dim=0)
        # Create initial tokens for all the beams. (n_beam, ctx)
        ctx_tokens = torch.tensor([[*self.tokenizer.sot_sequence]]).repeat_interleave(self.n_beam, dim=0)
        # Maintains the sum of log probabilities for uncompleted beams at each step.
        sum_logprobs = [0] * self.n_beam
        # Stores tokens of completed beams.
        final_beams_tokens = []
        # Stores log probabilities of completed beams. The probabilities are in the same order as tokens.
        final_beams_logprobs = []
        # Keeps track us how many beams are completed. Equivalent to `len(final_beams_tokens)`
        n_completed = 0
        for _ in range(self.max_sample_len):
            if n_completed == self.n_beam:
                break
            # Calculate the logits. TODO: Possible optimization because the audio features is always the same in each iteration.
            logits = self.inference.logits(ctx_tokens, audio_features)
            logits = logits[:,-1]
            self.suppress_logits(logits, ctx_tokens)
            logprobs = F.log_softmax(logits, dim=-1)
            # Select the top `n_beam` tokens from each uncompleted beam. Shape = (n_beam-n_completed, n_beam).
            top_logprobs, top_tokens = logprobs.topk(self.n_beam, dim=-1, largest=True, sorted=True)  
            # For each of the selected tokens, we concatenate the token with the prefix tokens from its beam
            # thus forming a new beam and calculate the beam's log probability. We then store them in the scores
            # dictionary where the keys are the new beams tokens and the values are the log probabilities.
            scores = {}
            for row_idx in range(self.n_beam - n_completed):
                for col_idx in range(self.n_beam):
                    cumulative_logprob = sum_logprobs[row_idx] + top_logprobs[row_idx, col_idx]
                    prefix = ctx_tokens[row_idx].tolist() + [top_tokens[row_idx, col_idx].item()]
                    # Storing the prefixes as keys prevents having two beams with the same tokens in
                    # the scores dictionary.
                    scores[tuple(prefix)] = cumulative_logprob
            new_ctx_tokens = []
            new_sum_logprobs = []
            # Sorts the prefixes by their corresponding sum of log probs and picks the top beams.
            for prefix in sorted(scores, key=scores.get, reverse=True)[:self.n_beam - n_completed]:
                logprob = scores[prefix]
                pred_token = prefix[-1]
                if pred_token == self.tokenizer.eot:
                    # We slice the prefix to remove the initial tokens and also the end-of-text tokens.
                    completed_beam_tokens = list(prefix)[len(self.tokenizer.sot_sequence):-1]
                    final_beams_tokens.append(completed_beam_tokens)
                    final_beams_logprobs.append(logprob)
                    n_completed = n_completed + 1

                    # Slices the audio features and kv cache to account for completed beams.
                    audio_features = audio_features[:self.n_beam - n_completed]
                    for module, cached in self.inference.cache.items():
                        self.inference.cache[module] = cached[:self.n_beam - n_completed]
                else:
                    new_ctx_tokens.append(list(prefix))
                    new_sum_logprobs.append(logprob)
            ctx_tokens = torch.tensor(new_ctx_tokens)
            sum_logprobs = new_sum_logprobs
        if final_beams_tokens:
            # Return highest probability sequence of the final `n_beam` sequences.
            # TODO: Add length normalization or length penalty.
            top_idx = final_beams_logprobs.index(max(final_beams_logprobs))
            out_tokens = list(final_beams_tokens[top_idx])
        else:
            # If none was completed, pick the beam that currently has the highest logprobs.
            top_idx = sum_logprobs.index(max(sum_logprobs))
            out_tokens = ctx_tokens[top_idx].tolist()[len(self.tokenizer.sot_sequence):]
        result = self.create_transcript(out_tokens)
        return result


class SamplingDecoder(Decoder):
    """Implements decoding by stochastic sampling.

    Stochastic sampling works by sampling the next token according to the probability distribution given by
    the model. It is fast, less prone to repetition than any other method but may provide incoherent
    transcriptions. Sampling is controlled by a temperature parameter in [0, 1]. The logits are divided
    by the temperature before they are normalized to form a probability distribution. The closer the
    temperature is to zero, the more the logits with high probability get 'boosted' and the logits with
    less probability get shrunk and vice-versa. Essentially, the behaviour of sampling decoder with very low 
    temperature approaches that of greedy decoder while higher temperature allows for diversity while risking
    incoherent output.
    """
    def __init__(self, model: Whisper, tokenizer: Tokenizer, temperature: float):
        super().__init__(model, tokenizer)

        self.temperature = temperature

    def _decode_segment(self, spectrogram):
        audio_features = self.inference.embed_audio(spectrogram)
        ctx_tokens = torch.tensor([[*self.tokenizer.sot_sequence]])
        out_tokens = []
        for i in range(self.max_sample_len):
            logits = self.inference.logits(ctx_tokens, audio_features)
            logits = logits[:,-1]
            self.suppress_logits(logits, ctx_tokens)
            logits = logits / self.temperature
            probs = F.softmax(logits, dim=-1)
            pred_token = torch.multinomial(probs, num_samples=1)
            if pred_token == self.tokenizer.eot:
                break
            ctx_tokens = torch.tensor([ctx_tokens[0].tolist() + [pred_token]])
            out_tokens.append(pred_token.item())
        result = self.create_transcript(out_tokens)
        return result


def detect_language(spectrogram: Tensor, model: Whisper) -> str:
    """Detects the language spoken in the given audio spectrogram.

    Args:
        spectrogram: A tensor of shape (n_mels, n_frames).
        model: Whisper model instance.

    Returns:
        A string representing language id. for instance 'en' for English, or 'es' Spanish.
    """
    audio_segment = pad_or_trim_spectrogram(spectrogram)
    audio_features = model.embed_audio(audio_segment)
    tokenizer = get_tokenizer(multilingual=True)
    tokens = torch.tensor([[tokenizer.sot]]).to(audio_features.device) 
    logits = model.logits(tokens, audio_features)[:, 0]
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_token_probs = logits.softmax(dim=-1).cpu()
    language_token = language_token_probs.argmax(dim=-1)
    language_id = tokenizer.decode(language_token).split('|')[1]
    return language_id


def format_timestamp(timestamp: float) -> str:
    milliseconds = round(timestamp * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def save_to_srt(segment_transcripts: List[AudioSegmentTranscript], tokenizer: Tokenizer, fname: str) -> None:
    """Decodes the given audio transcription tokens and saves them in srt format.
    
    Args:
        batch_tokens: A list of lists where each inner list at pos i contains transcription tokens of 30-second
         audio segment at pos i in original audio. The inner lists also contains inner lists which contains
         timestamped segments of text. [ [ [<0.00>, text..., <5.00>]... ]... ] when tokens are decoded.
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
                start_ts = float(tokenizer.decode_with_timestamps([ts_tokens.pop(0)]).split('|')[1])
                start_time = format_timestamp(start_ts + segment_offset)
                end_ts = float(tokenizer.decode_with_timestamps([ts_tokens.pop(-1)]).split('|')[1])
                # Don't allow a segment final timestamp to be greater than 30.0 secs.
                end_ts = end_ts if end_ts <= 30.0 else 30.0
                end_time = format_timestamp(end_ts + segment_offset)
                timestamp = f'{start_time} --> {end_time}'
                text = tokenizer.decode(ts_tokens).lstrip()
                srt_segment = f'{index}\n{timestamp}\n{text}\n\n'
                f.write(srt_segment)
                ts_tokens_ctr += 1
            segment_offset += end_ts
    print('\nSrt subtitles successfully saved.')

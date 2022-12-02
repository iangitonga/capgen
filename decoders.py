"""Handles Text processing tasks."""

import abc
import time
from collections.abc import Sequence

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import List, Tuple, Union

from audio import pad_or_trim_spectrogram, N_FRAMES, N_FRAMES_PER_SECOND
from tokenizer import Tokenizer

# CONSTANTS
AVAILABLE_DECODERS = ('greedy', 'beamsearch',)


class Decoder(abc.ABC):
    def __init__(self, model, tokenizer, max_sample_len):
        self.model = model
        self.tokenizer = tokenizer
        self.suppress_tokens = self._get_suppress_tokens()
        self.max_sample_len = max_sample_len

    def segment_tokens(self, tokens: List[Tensor]) -> List[List[Tensor]]:
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

    def suppress_logits(self, logits: Tensor, tokens: Tensor) -> None:
        # TODO: improve suppression of repeating grams.
        self._suppress_repeating_grams(logits, tokens)
        self._suppress_tokens(logits, tokens)
        self._suppress_blank(logits, tokens)
        self._apply_timestamp_rules(logits, tokens)
    
    def _suppress_blank(self, logits: Tensor, tokens: Tensor) -> None:
        if tokens.shape[1] == 1:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
            
    def _suppress_tokens(self, logits: Tensor, tokens: Tensor) -> Tensor:
        logits[:, self.suppress_tokens] = -np.inf

    def _suppress_repeating_grams(self, logits: Tensor, tokens: Tensor) -> None:
        # A hack to suppress repetition of two tokens for more than five times.
        # Should be replaced with a better technique.
        n_beam, n_ctx = tokens.shape
        if n_ctx < 6:
            return
        for i in range(n_beam):
            gram1, gram2 = tokens[i][-2:]
            # suppress repeating token.
            if torch.all(tokens[i][-5:] == gram2):
                logits[:, gram2] = -np.inf
             # suppress two repeating tokens.
            if torch.all(tokens[i][-9::2] == gram2):
                logits[:, gram1] = -np.inf
            if torch.all(tokens[i][-10::2] == gram1):
                logits[:, gram2] = -np.inf
                
                
        
    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = []
        suppress_tokens.extend(self.tokenizer.non_speech_tokens)

        suppress_tokens.extend([self.tokenizer.sot, self.tokenizer.sot_prev, self.tokenizer.sot_lm])
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))
    
    def _apply_timestamp_rules(self, logits: Tensor, tokens: Tensor):
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
            sample_begin = 1  # len of initial tokens.
            seq = [t for t in tokens[k, sample_begin :].tolist()]
            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin

            if last_was_timestamp:
                # timestamps have appeared in pairs.
                if penultimate_was_timestamp:  # has to be non-timestamp
                    logits[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    logits[k, : self.tokenizer.eot] = -np.inf

        # apply the `max_initial_timestamp` option
        if tokens.shape[1] == sample_begin and max_initial_timestamp_index is not None:
            last_allowed = self.tokenizer.timestamp_begin + max_initial_timestamp_index
            logits[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = F.log_softmax(logits.float(), dim=-1)
        for k in range(tokens.shape[0]):
            timestamp_logprob = logprobs[k, self.tokenizer.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.tokenizer.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob:
                logits[k, : self.tokenizer.timestamp_begin] = -np.inf


class BeamSearchDecoder(Decoder):
    @torch.no_grad()
    def _decode(self, audio_features: Tensor, n_beam: int) -> List[List[Tensor]]:
        """Extracts tokens for the given audio features using beamsearch of the given width.
        
        Args:
            audio_features: A tensor of shape (1, n_audio_ctx, n_audio_state).
            n_beam: Number of beams to search at every iteration.
        
        Returns:
            A list of lists where each list contains a start timestamp token, text tokens and
             end timestamp tokens. The inner lists are ordered so that the timestamped segments
             appear in the order they appeared in the audio.
        """
        # Copy the audio features to use for each beam. (n_beam, T, D)
        audio_features = audio_features.repeat_interleave(n_beam, dim=0)
        # Create initial tokens. (n_beam, ctx)
        tokens = torch.tensor([[self.tokenizer.sot]]).repeat_interleave(n_beam, dim=0)
        sum_logprobs = [0] * n_beam
        completed_sequences = []
        completed_logprobs = []
        n_completed = 0
        for i in range(self.max_sample_len):
            print(f'Decoder counter: {i}/{self.max_sample_len}', end='\r')
            if n_completed == n_beam:
                print('<>', end=' ')
                break
            # Calculate the logits and select the top `n_beam` beams.
            logits = self.model.logits(tokens, audio_features)
            logits = logits[:,-1]
            self.suppress_logits(logits, tokens)
            logprobs = F.log_softmax(logits, dim=-1)
            top_logprobs, top_tokens = logprobs.topk(n_beam, dim=-1, largest=True, sorted=True)  # (n_beam-n_completed, n_beam)
            # Organise the the log probabilities, contexts and scores for sorting and selection.
            scores = {}
            for row_idx in range(n_beam - n_completed):
                for col_idx in range(n_beam):
                    cum_logprob = sum_logprobs[row_idx] + top_logprobs[row_idx, col_idx]
                    prefix = tokens[row_idx].tolist() + [top_tokens[row_idx, col_idx].item()]
                    # This is a hack that helps us avoid base case. In the first iteration, we have
                    # the same audio segment and tokens in all beams. Thus, if we did a naive sorting,
                    # of top tokens, the sorting would result in the highest prob token being selected
                    # for all the beams since the top prob token is the same across all the beams.
                    scores[tuple(prefix)] = cum_logprob
            # Select the top transcriptions so far.
            new_tokens = []
            new_sum_logprobs = []
            for prefix in sorted(scores, key=scores.get, reverse=True)[:n_beam - n_completed]:
                logprob = scores[prefix]
                pred_token = prefix[-1]
                if pred_token == self.tokenizer.eot:
                    completed_sequences.append(list(prefix))
                    completed_logprobs.append(logprob)
                    n_completed = n_completed + 1
                else:
                    new_tokens.append(list(prefix))
                    new_sum_logprobs.append(logprob)
            tokens = torch.tensor(new_tokens)
            sum_logprobs = new_sum_logprobs
            audio_features = audio_features[:n_beam - n_completed]
        if completed_sequences:
            # Return highest probability sequence of the final `n_beam` sequences.
            top_idx = completed_logprobs.index(max(completed_logprobs))
            final_tokens = list(completed_sequences[top_idx])
        else:
            top_idx = sum_logprobs.index(max(sum_logprobs))
            final_tokens = tokens[top_idx].tolist()
        final_tokens.pop(0) # Remove <sot> and <eot> tokens.
        final_tokens.pop(-1)
        final_tokens = self.segment_tokens(final_tokens)
        return final_tokens
    
    @torch.no_grad()
    def decode(self, spectrogram: Tensor, n_beam: int) -> List[List[List[Tensor]]]:
        """Extracts transcription tokens of given mel spectrogram using beam search.
        
        Beam search works by keeping the most probable `n_beam` number of transcriptions at each timestep. The
        probability of a transcription is determined by the product of the probabilities of each individual token
        or equivalently the sum of log probabilities of each token(avoids risk of underflow).

        Args:
            spectrogram: The spectrogram of an entire audio with shape (n_segments, n_mels, 3000).
        Returns:
            A list of lists of length n_spectrogram where each inner list contains the token transcriptions of
             the corresponding segment.
        """
        print('Decoding ...')
        _, T = spectrogram.shape
        # An addition by 1 accounts for overlapping of segments and where `T % N_FRAMES` is not zero.
        n_spectrogram =( T // N_FRAMES) + 1
        # Offset position tells us the starting frame position of the next segment. The starting position of the
        # next segment starts where the previous segment generated the final timestamp. This allows overlapping
        # of segments which ensures that every section of audio is properly transcribed.
        offset_pos = 0
        transcription_tokens = []
        for i in range(n_spectrogram):
            t1 = time.perf_counter()

            seg_spectrogram = pad_or_trim_spectrogram(spectrogram[:,offset_pos:]).unsqueeze(0)
            seg_spectrogram_features = self.model.embed_audio(seg_spectrogram)
            seg_spectrogram_tokens = self._decode(seg_spectrogram_features, n_beam)
            transcription_tokens.append(seg_spectrogram_tokens)
            last_ts = self.tokenizer.decode_with_timestamps([seg_spectrogram_tokens[-1][-1]]).split('|')[1]
            prev_segment_ts_delta = 30.0 - float(last_ts)
            offset_pos += int(N_FRAMES - N_FRAMES_PER_SECOND * prev_segment_ts_delta)

            time_delta = time.perf_counter() - t1
            print(f'COMPLETED SEGMENT {i+1}/{n_spectrogram}, {time_delta:.3f} secs.')
        return transcription_tokens



def format_timestamp(tokenizer: Tokenizer, ts_token: Union[int, Tensor], offset: float) -> str:
    time_string = tokenizer.decode_with_timestamps([ts_token]).split('|')[1]
    seconds = float(time_string) + offset
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def save_to_srt(batch_tokens: List[List[List[Union[int, Tensor]]]], tokenizer: Tensor, fname: str) -> None:
    """Decodes the given audio transcription tokens and saves them in srt format.
    
    Args:
        batch_tokens: A list of lists where each inner list at pos i contains transcription tokens of 30-second
         audio segment at pos i in original audio. The inner lists also contains inner lists which contains
         timestamped segments of text. [ [ [<0.00>, text..., <5.00>]... ]... ] when tokens are decoded.
        tokenizer: A tokenizer to decode the tokens.
        fname: The name of the file to save the tokens.
    """
    print('Saving ...', end='')
    with open(fname, 'w') as f:
        ts_tokens_ctr = 1
        segment_offset = 0
        for i in range(len(batch_tokens)):
            segment_tokens = batch_tokens[i]
            for ts_tokens in segment_tokens:
                index = str(ts_tokens_ctr)
                start_ts = format_timestamp(tokenizer, ts_tokens.pop(0), segment_offset)
                end_ts_token = ts_tokens.pop(-1)
                end_ts = format_timestamp(tokenizer, end_ts_token, segment_offset)
                timestamp = f'{start_ts} --> {end_ts}'
                text = tokenizer.decode(ts_tokens).lstrip()
                srt_segment = f'{index}\n{timestamp}\n{text}\n\n'
                f.write(srt_segment)
                ts_tokens_ctr += 1
            segment_offset += float(tokenizer.decode_with_timestamps([end_ts_token]).split('|')[1])
    print('Done')
    return

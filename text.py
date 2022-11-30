"""Handles Text processing tasks."""

import abc
import time

import torch
import torch.nn.functional as F
import numpy as np



class Decoder(abc.ABC):
    def __init__(self, model, tokenizer, max_sample_len):
        self.model = model
        self.tokenizer = tokenizer
        self.suppress_tokens = self._get_suppress_tokens()
        self.max_sample_len = max_sample_len

    def suppress_logits(self, logits, tokens):
        self._suppress_tokens(logits, tokens)
        self._suppress_blank(logits, tokens)
        self._apply_timestamp_rules(logits, tokens)
    
    def _suppress_blank(self, logits, tokens):
        if tokens.shape[1] == 1:
            logits[:, self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf
            
    def _suppress_tokens(self, logits, tokens):
        logits[:, self.suppress_tokens] = -np.inf
        
    def _get_suppress_tokens(self):
        suppress_tokens = []
        suppress_tokens.extend(self.tokenizer.non_speech_tokens)

        suppress_tokens.extend([self.tokenizer.sot, self.tokenizer.sot_prev, self.tokenizer.sot_lm])
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))
    
    def _apply_timestamp_rules(self, logits, tokens):
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


# TODO: Replace decoder with correct version.
class BeamSearchDecoder(Decoder):
    @torch.no_grad()
    def _decode(self, audio_features, n_beam):
        # Copy the audio features to use for each beam. (n_beam, T, D)
        audio_features = audio_features.repeat_interleave(n_beam, dim=0)
        # Create initial tokens. (n_beam, ctx)
        tokens = torch.tensor([[self.tokenizer.sot]]*n_beam)
        sum_logprobs = [0]*n_beam
        completed_sequences = []
        n_completed = 0
        for i in range(self.max_sample_len):
            if n_completed == n_beam:
                print('<>', end=' ')
                break
            # Calculate the logits and select the top `n_beam` beams.
            logits = self.model.logits(tokens, audio_features)
            logits = logits[:,-1]
            self.suppress_logits(logits, tokens)
            logprobs = F.log_softmax(logits, dim=-1)
            top_logprobs, top_tokens = logprobs.topk(n_beam, dim=-1, largest=True, sorted=True)  # (n_beam, n_beam)
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
                    completed_sequences.append(prefix)
                    n_completed = n_completed + 1
                else:
                    new_tokens.append(list(prefix))
                    new_sum_logprobs.append(logprob)
            tokens = torch.tensor(new_tokens)
            sum_logprobs = new_sum_logprobs
            audio_features = audio_features[:n_beam - n_completed]
        if completed_sequences:
            # Return highest probability sequence of the final `n_beam` sequences.
            final_tokens = list(completed_sequences[0])
            final_tokens.pop(0) # Remove <sot> and <eot> tokens.
            final_tokens.pop(-1)
            return final_tokens
        highest_idx = sum_logprobs.index(max(sum_logprobs))
        highest = tokens[highest_idx].tolist()
        return highest
    
    @torch.no_grad()
    def decode(self, mel_spectogram, n_beam):
        """Extracts transcription tokens of given mel spectogram using beam search.
        
        Beam search works by keeping the most probable `n_beam` number of transcriptions at each timestep. The
        probability of a transcription is determined by the product of the probabilities of each individual token
        or equivalently the sum of log probabilities of each token(avoids risk of underflow).

        Args:
            mel_spectogram: The spectogram of an entire audio with shape (n_chunks, N_MELS, CHUNK_LENGTH).
        Returns:
            A list of lists of length n_spectogram where each inner list contains the token transcriptions of
             the corresponding chunk.
        """
        n_chunks = mel_spectogram.shape[0]
        audio_features = self.model.embed_audio(mel_spectogram)
        transcription_tokens = []
        for i in range(n_chunks):
            t1 = time.perf_counter()

            feature = audio_features[i].unsqueeze(0)
            trans = self._decode(feature, n_beam)
            transcription_tokens.append(trans)

            delta = time.perf_counter() - t1
            print(f'COMPLETED SEGMENT {i+1}/{n_chunks}, {delta:.3f} secs.')
        return transcription_tokens


class GreedyDecoder(Decoder):
    def _decode(self, audio_features):
        # This function assumes that for each audio segment transcription tokens, there is a start timestamp,
        # an end timestamp and  non-starting and non-ending timestamp tokens always appear in pairs. This
        # functionality is implemented by token suppressors.
        # If the decoding reaches max_sample_length without encountering <eot> token, the last timestamped
        # tokens are dropped because they do not include the <eot> token.
        tokens_raw = [[self.tokenizer.sot]]
        # Contains lists of tokens where in each list, the first and the last tokens are the timestamps and
        # the rest of the tokens are text tokens.
        segment_tokens = []
        # Contains the current timestamped tokens.
        current_ts_tokens = []
        for i in range(self.max_sample_len):
            print(f'decoder counter: {i}/{self.max_sample_len}', end='\r')
            tokens = torch.tensor(tokens_raw)
            logits = self.model.decoder(tokens, audio_features)
            # Pluck out the logits for the current token.
            logits = logits[:,-1]
            self.suppress_logits(logits, tokens)
            logits = F.softmax(logits, dim=-1)
            next_token = logits.flatten().argmax()
            if next_token == self.tokenizer.eot:
                print('<>', end=' ')
                # In order to include the last timestamped transcription, it needs to have at least three tokens
                # where the tokens could potentially be two timestamps and one text token. Other-wise we do not
                # include it. 
                if len(current_ts_tokens) > 3:
                    segment_tokens.append(current_ts_tokens)
                break
            tokens_raw[0].append(next_token)
            
            current_ts_tokens.append(next_token)
            if next_token >= self.tokenizer.timestamp_begin and len(current_ts_tokens) > 2:
                segment_tokens.append(current_ts_tokens)
                current_ts_tokens = []
        return segment_tokens

    @torch.no_grad()
    def decode(self, mel_spectogram):
        """Extracts transcription tokens of given mel spectogram using a greedy strategy.
        
        A greedy decoder picks the most likely token predicted by the model at each timestep. It
        is however important to note that the final transcription predicted using this strategy is
        not necessarily the most probable. In most cases beam search offers better results.

        Args:
            mel_spectogram: The spectogram of an entire audio with shape (n_spectogram, N_MELS, CHUNK_LENGTH).
        Returns:
            A list of lists of length n_spectogram where each inner list contains the token transcriptions of
             the corresponding chunk.
        """
        n_spectogram = mel_spectogram.shape[0]
        audio_features = self.model.embed_audio(mel_spectogram)
        transcription_tokens = []
        for i in range(n_spectogram):
            t1 = time.perf_counter()

            feature = audio_features[i].unsqueeze(0)
            trans = self._decode(feature)
            transcription_tokens.append(trans)

            delta = time.perf_counter() - t1
            print(f'COMPLETED SEGMENT {i+1}/{n_spectogram}, {delta:.3f} secs.')
        return transcription_tokens



def format_timestamp(tokenizer, ts_token, offset):
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


def save_to_srt(batch_tokens, tokenizer, fname):
    print('Saving ...', end='')
    with open(fname, 'w') as f:
        ts_tokens_ctr = 1
        for i in range(len(batch_tokens)):
            segment_tokens = batch_tokens[i]
            ts_offset = i*30.0
            for ts_tokens in segment_tokens:
                index = str(ts_tokens_ctr)
                start_ts = format_timestamp(tokenizer, ts_tokens.pop(0), ts_offset)
                end_ts = format_timestamp(tokenizer, ts_tokens.pop(-1), ts_offset)
                timestamp = f'{start_ts} --> {end_ts}'
                text = tokenizer.decode(ts_tokens).lstrip()
                srt_segment = f'{index}\n{timestamp}\n{text}\n\n'
                f.write(srt_segment)
                ts_tokens_ctr += 1
    print('Done')
    return

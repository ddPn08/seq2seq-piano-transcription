import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import T5Config, T5ForConditionalGeneration

from .constants import AUDIO_SEGMENT_SEC


def split_audio_into_segments(y: torch.Tensor, sr: int):
    audio_segment_samples = round(AUDIO_SEGMENT_SEC * sr)
    pad_size = audio_segment_samples - (y.shape[-1] % audio_segment_samples)
    y = F.pad(y, (0, pad_size))
    assert (y.shape[-1] % audio_segment_samples) == 0
    n_chunks = y.shape[-1] // audio_segment_samples
    y_segments = torch.chunk(y, chunks=n_chunks, dim=-1)
    return torch.stack(y_segments, dim=0)


def unpack_sequence(x: torch.Tensor, eos_id: int = 1):
    seqs = []
    max_length = x.shape[-1]
    for seq in x:
        start_pos = 0
        pos = 0
        while (pos < max_length) and (seq[pos] != eos_id):
            pos += 1
        end_pos = pos + 1
        seqs.append(seq[start_pos:end_pos])
    return seqs


class LogMelspec(nn.Module):
    def __init__(self, sample_rate, n_fft, n_mels, hop_length):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=20.0,
            n_mels=n_mels,
            mel_scale="slaney",
            norm="slaney",
            power=1,
        )
        self.eps = 1e-5

    def forward(self, x):
        spec = self.melspec(x)
        safe_spec = torch.clamp(spec, min=self.eps)
        log_spec = torch.log(safe_spec)
        return log_spec


class Seq2SeqTranscriber(nn.Module):
    def __init__(
        self, n_mels: int, sample_rate: int, n_fft: int, hop_length: int, voc_dict: dict
    ):
        super().__init__()
        self.infer_max_len = 200
        self.voc_dict = voc_dict
        self.n_voc_token = voc_dict["n_voc"]
        self.t5config = T5Config.from_pretrained("google/t5-v1_1-small")
        custom_configs = {
            "vocab_size": self.n_voc_token,
            "pad_token_id": voc_dict["pad"],
            "d_model": n_mels,
        }

        for k, v in custom_configs.items():
            self.t5config.__setattr__(k, v)

        self.transformer = T5ForConditionalGeneration(self.t5config)
        self.melspec = LogMelspec(sample_rate, n_fft, n_mels, hop_length)
        self.sr = sample_rate

    def forward(self, wav, labels):
        spec = self.melspec(wav).transpose(-1, -2)
        outs = self.transformer.forward(
            inputs_embeds=spec, return_dict=True, labels=labels
        )
        return outs

    def infer(self, wav):
        """
        Infer the transcription of a single audio file.
        The input audio file is split into segments of 2 seconds
        before passing to the transformer.
        """
        wav_segs = split_audio_into_segments(wav, self.sr)
        spec = self.melspec(wav_segs).transpose(-1, -2)
        outs = self.transformer.generate(
            inputs_embeds=spec,
            max_length=self.infer_max_len,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=False,
        )
        return outs

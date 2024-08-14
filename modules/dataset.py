import numpy as np
import torch
import torch.utils.data as data
import torchaudio
import tqdm

from .constants import (
    AUDIO_SEGMENT_SEC,
    FRAME_PER_SEC,
    FRAME_STEP_SIZE_SEC,
    SEGMENT_N_FRAMES,
)
from .tokenizer import MIDITokenExtractor


class AMTDatasetBase(data.Dataset):
    def __init__(
        self,
        flist_audio,
        flist_midi,
        sample_rate,
        voc_dict,
        apply_pedal=False,
        whole_song=False,
        cache_in_memory=False,
    ):
        super().__init__()
        self.midi_filelist = flist_midi
        self.audio_filelist = flist_audio
        self.voc_dict = voc_dict
        self.midi_list = [
            MIDITokenExtractor(f, voc_dict, apply_pedal)
            for f in tqdm.tqdm(self.midi_filelist, desc="load dataset")
        ]
        self.sample_rate = sample_rate
        self.whole_song = whole_song

        self.cache_in_memory = cache_in_memory
        if cache_in_memory:
            self.audio_list = []
            for f in tqdm.tqdm(flist_audio, desc="cache audio"):
                wav, sr = torchaudio.load(f)
                wav = wav.mean(0)
                if sr != sample_rate:
                    wav = torchaudio.functional.resample(
                        wav,
                        sr,
                        sample_rate,
                        resampling_method="sinc_interp_kaiser",
                    )

                self.audio_list.append(wav)

        self.audio_metalist = [torchaudio.info(f) for f in flist_audio]

    def __len__(self):
        return len(self.audio_filelist)

    def __getitem__(self, index):
        """
        Return a pair of (audio, tokens) for the given index.
        On the training stage, return a random segment from the song.
        On the test stage, return the audio and MIDI of the whole song.
        """
        if not self.whole_song:
            return self.getitem_segment(index)
        else:
            return self.getitem_wholesong(index)

    def getitem_segment(self, index, start_pos=None):
        metadata = self.audio_metalist[index]
        num_frames = metadata.num_frames
        sample_rate = metadata.sample_rate
        duration_y = round(num_frames / float(sample_rate) * FRAME_PER_SEC)
        midi_item: MIDITokenExtractor = self.midi_list[index]
        if start_pos is None:
            segment_start = np.random.randint(duration_y - SEGMENT_N_FRAMES)
        else:
            segment_start = start_pos
        segment_end = segment_start + SEGMENT_N_FRAMES
        segment_start_sample = round(segment_start * FRAME_STEP_SIZE_SEC * sample_rate)

        segment_tokens = midi_item.get_segment_tokens(segment_start, segment_end)
        segment_tokens = torch.from_numpy(segment_tokens).long()
        if self.cache_in_memory:
            y, _ = self.audio_list[index]
            y_segment = y[:, segment_start:segment_end]
        else:
            y_segment, _ = torchaudio.load(
                self.audio_filelist[index],
                frame_offset=segment_start_sample,
                num_frames=round(AUDIO_SEGMENT_SEC * sample_rate),
            )
            y_segment = y_segment.mean(0)
            if sample_rate != self.sample_rate:
                y_segment = torchaudio.functional.resample(
                    y_segment,
                    sample_rate,
                    self.sample_rate,
                    resampling_method="sinc_interp_kaiser",
                )

        return y_segment, segment_tokens

    def getitem_wholesong(self, index):
        y, sr = torchaudio.load(self.audio_filelist[index])
        y = y.mean(0)
        if sr != self.sample_rate:
            y = torchaudio.functional.resample(
                y, sr, self.sample_rate, resampling_method="sinc_interp_kaiser"
            )
        midi = self.midi_list[index].pm
        return y, midi

    def collate_wholesong(self, batch):
        batch_audio = torch.stack([b[0] for b in batch], dim=0)
        midi = [b[1] for b in batch]
        return batch_audio, midi

    def collate_batch(self, batch):
        batch_audio = torch.stack([b[0] for b in batch], dim=0)
        batch_tokens = [b[1] for b in batch]
        batch_tokens_pad = torch.nn.utils.rnn.pad_sequence(
            batch_tokens, batch_first=True, padding_value=self.voc_dict["pad"]
        )
        return batch_audio, batch_tokens_pad

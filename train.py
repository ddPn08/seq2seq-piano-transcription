import os

import pandas as pd
import torch
import torch.utils.data as data
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from modules.constants import SAMPLING_RATE
from modules.dataset import AMTDatasetBase
from modules.evaluate import evaluate_midi
from modules.tokenizer import token_seg_list_to_midi, voc_single_track
from modules.transcriber import Seq2SeqTranscriber, unpack_sequence


class Maestro(AMTDatasetBase):
    def __init__(
        self,
        n_files=-1,
        sample_rate=44100,
        split="test",
        apply_pedal=False,
        whole_song=False,
    ):
        data_path = "/root/maestro-v3.0.0/"
        df_metadata = pd.read_csv(os.path.join(data_path, "maestro-v3.0.0.csv"))
        flist_audio = []
        flist_midi = []
        list_title = []
        for row in range(len(df_metadata)):
            if df_metadata["split"][row] == split:
                f_audio = os.path.join(data_path, df_metadata["audio_filename"][row])
                f_midi = os.path.join(data_path, df_metadata["midi_filename"][row])
                assert os.path.exists(f_audio) and os.path.exists(f_midi)
                flist_audio.append(f_audio)
                flist_midi.append(f_midi)
                list_title.append(df_metadata["canonical_title"][row])
        if n_files > 0:
            flist_audio = flist_audio[:n_files]
            flist_midi = flist_midi[:n_files]
        super().__init__(
            flist_audio,
            flist_midi,
            sample_rate,
            voc_dict=voc_single_track,
            apply_pedal=apply_pedal,
            whole_song=whole_song,
        )
        self.list_title = list_title


class LitTranscriber(LightningModule):
    def __init__(
        self,
        transcriber_args: dict,
        lr: float,
        lr_decay: float = 1.0,
        lr_decay_interval: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.voc_dict = voc_single_track
        self.n_voc = self.voc_dict["n_voc"]
        self.transcriber = Seq2SeqTranscriber(
            **transcriber_args, voc_dict=self.voc_dict
        )
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval

    def forward(self, y: torch.Tensor):
        transcriber_infer = self.transcriber.infer(y)
        return transcriber_infer

    def training_step(self, batch, batch_idx):
        y, t = batch
        tf_out = self.transcriber(y, t)
        loss = tf_out.loss
        t = t.detach()
        mask = t != self.voc_dict["pad"]
        accr = (tf_out.logits.argmax(-1)[mask] == t[mask]).sum() / mask.sum()
        self.log("train/loss", loss)
        self.log("train/accr", accr)
        return loss

    def validation_step(self, batch, batch_idx):
        assert not self.transcriber.training
        y, t = batch
        tf_out = self.transcriber(y, t)
        loss = tf_out.loss
        t = t.detach()
        mask = t != self.voc_dict["pad"]
        accr = (tf_out.logits.argmax(-1)[mask] == t[mask]).sum() / mask.sum()
        self.log("vali/loss", loss)
        self.log("vali/accr", accr)
        return loss

    def test_step(self, batch, batch_idx):
        y, ref_midi = batch
        y = y[0]
        ref_midi = ref_midi[0]
        with torch.no_grad():
            est_tokens = self.forward(y)
            unpadded_tokens = unpack_sequence(est_tokens.cpu().numpy())
            unpadded_tokens = [t[1:] for t in unpadded_tokens]
            est_midi = token_seg_list_to_midi(unpadded_tokens)
        dict_eval = evaluate_midi(est_midi, ref_midi)
        dict_log = {}
        for key in dict_eval:
            dict_log["test/" + key] = dict_eval[key]
        self.log_dict(dict_log, batch_size=1)

    def train_dataloader(self):
        dset = Maestro(
            sample_rate=SAMPLING_RATE,
            split="train",
            n_files=2000,
        )
        return data.DataLoader(
            dataset=dset,
            collate_fn=dset.collate_batch,
            batch_size=64,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
        )

    def test_dataloader(self):
        dset = Maestro(
            sample_rate=SAMPLING_RATE,
            split="test",
            whole_song=True,
            n_files=10,
        )
        return data.DataLoader(
            dataset=dset,
            collate_fn=dset.collate_wholesong,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main():
    args = {
        "n_mels": 512,
        "sample_rate": 16000,
        "n_fft": 2048,
        "hop_length": 128,
    }

    lightning_module = LitTranscriber(transcriber_args=args, lr=1e-4, lr_decay=0.99)

    logger = WandbLogger(project="seq2seq-piano-transcription", name="test-01")
    checkpoint = ModelCheckpoint(
        dirpath="output/checkpoints",
        filename="{epoch}",
        every_n_epochs=50,
        save_on_train_epoch_end=True,
    )

    trainer = Trainer(
        logger=logger,
        accelerator="gpu",
        devices="0,",
        max_epochs=10000,
        callbacks=[checkpoint],
    )

    trainer.fit(lightning_module)


if __name__ == "__main__":
    main()

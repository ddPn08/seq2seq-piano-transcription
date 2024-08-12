import argparse

import torch
import torchaudio

from modules.constants import SAMPLING_RATE
from modules.tokenizer import token_seg_list_to_midi, voc_single_track
from modules.transcriber import Seq2SeqTranscriber


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("output", type=str, help="Output file")
    parser.add_argument("--model", type=str, help="Model path", default="model.pt")
    parser.add_argument("--device", type=str, help="Device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    audio, sr = torchaudio.load(args.input)
    if sr != SAMPLING_RATE:
        audio = torchaudio.functional.resample(
            audio, sr, SAMPLING_RATE, resampling_method="sinc_interp_kaiser"
        )
    audio = audio.mean(0)

    transcriber = Seq2SeqTranscriber(
        n_mels=512,
        sample_rate=SAMPLING_RATE,
        n_fft=2048,
        hop_length=128,
        voc_dict=voc_single_track,
    )
    state_dict: dict = torch.load(args.model, map_location="cpu")

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any(k.startswith("transcriber.") for k in state_dict):
        state_dict = {
            k.replace("transcriber.", ""): v
            for k, v in state_dict.items()
            if "transcriber." in k
        }

    transcriber.load_state_dict(state_dict)
    transcriber.eval()

    transcriber.to(args.device)
    audio = audio.to(args.device)

    with torch.no_grad():
        outputs: torch.LongTensor = transcriber.infer(audio)
        mid = token_seg_list_to_midi(outputs.cpu().numpy())
        mid.write(args.output)


if __name__ == "__main__":
    main()

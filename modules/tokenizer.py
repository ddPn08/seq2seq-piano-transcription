import dataclasses
from operator import attrgetter

import numpy as np
import pretty_midi as pm
import torch

from .constants import AUDIO_SEGMENT_SEC, FRAME_PER_SEC, FRAME_STEP_SIZE_SEC

N_NOTE = 128
N_TIME = 205
N_SPECIAL = 3

voc_single_track = {
    "pad": 0,
    "eos": 1,
    "endtie": 2,
    "note": N_SPECIAL,
    "onset": N_SPECIAL + N_NOTE,
    "time": N_SPECIAL + N_NOTE + 2,
    "n_voc": N_SPECIAL + N_NOTE + 2 + N_TIME + 3,
    "keylist": ["pad", "eos", "endtie", "note", "onset", "time"],
}


@dataclasses.dataclass
class Event:
    prog: int
    onset: bool
    pitch: int


class MIDITokenExtractor:
    def __init__(self, midi_path: str, voc_dict: dict, apply_pedal: bool = False):
        self.pm = pm.PrettyMIDI(midi_path)
        if apply_pedal:
            self.pm_apply_pedal(self.pm)
        self.voc_dict = voc_dict
        self.multi_track = "instrument" in voc_dict

    def pm_apply_pedal(self, pm: pm.PrettyMIDI, program: int = 0):
        """
        Apply sustain pedal by stretching the notes in the pm object.
        """
        # 1: Record the onset positions of each notes as a dictionary
        onset_dict = dict()
        for note in pm.instruments[program].notes:
            if note.pitch in onset_dict:
                onset_dict[note.pitch].append(note.start)
            else:
                onset_dict[note.pitch] = [note.start]
        for k in onset_dict.keys():
            onset_dict[k] = np.sort(onset_dict[k])

        # 2: Record the pedal on/off state of each time frame
        arr_pedal = np.zeros(round(pm.get_end_time() * FRAME_PER_SEC) + 100, dtype=bool)
        pedal_on_time = -1
        list_pedaloff_time = []
        for cc in pm.instruments[program].control_changes:
            if cc.number == 64:
                if (cc.value > 0) and (pedal_on_time < 0):
                    pedal_on_time = round(cc.time * FRAME_PER_SEC)
                elif (cc.value == 0) and (pedal_on_time >= 0):
                    pedal_off_time = round(cc.time * FRAME_PER_SEC)
                    arr_pedal[pedal_on_time:pedal_off_time] = True
                    list_pedaloff_time.append(cc.time)
                    pedal_on_time = -1
        list_pedaloff_time = np.sort(list_pedaloff_time)

        # 3: Stretch the notes (modify note.end)
        for note in pm.instruments[program].notes:
            # 3-1: Determine whether sustain pedal is on at note.end. If not, do nothing.
            # 3-2: Find the next note onset time and next pedal off time after note.end.
            # 3-3: Extend note.end till the minimum of next_onset and next_pedaloff.
            note_off_frame = round(note.end * FRAME_PER_SEC)
            pitch = note.pitch
            if arr_pedal[note_off_frame]:
                next_onset = np.argwhere(onset_dict[pitch] > note.end)
                next_onset = (
                    np.inf
                    if len(next_onset) == 0
                    else onset_dict[pitch][next_onset[0, 0]]
                )
                next_pedaloff = np.argwhere(list_pedaloff_time > note.end)
                next_pedaloff = (
                    np.inf
                    if len(next_pedaloff) == 0
                    else list_pedaloff_time[next_pedaloff[0, 0]]
                )
                new_noteoff_time = max(note.end, min(next_onset, next_pedaloff))
                new_noteoff_time = min(new_noteoff_time, pm.get_end_time())
                note.end = new_noteoff_time

    def get_segment_tokens(self, start: float, end: float):
        """
        Transform a segment of the MIDI file into a sequence of tokens.
        """
        dict_event = dict()  # a dictionary that maps time to a list of events.

        def append_to_dict_event(time, item):
            if time in dict_event:
                dict_event[time].append(item)
            else:
                dict_event[time] = [item]

        list_events = []  # events section
        list_tie_section = []  # tie section

        for instrument in self.pm.instruments:
            prog = instrument.program
            for note in instrument.notes:
                note_end = round(note.end * FRAME_PER_SEC)
                note_start = round(note.start * FRAME_PER_SEC)
                if (note_end < start) or (note_start >= end):
                    continue
                if (note_start < start) and (note_end >= start):
                    # If the note starts before the segment, but ends in the segment
                    # it is added to the tie section.
                    list_tie_section.append(self.voc_dict["note"] + note.pitch)
                    if note_end < end:
                        append_to_dict_event(
                            note_end - start, Event(prog, False, note.pitch)
                        )
                    continue
                assert note_start >= start
                append_to_dict_event(note_start - start, Event(prog, True, note.pitch))
                if note_end < end:
                    append_to_dict_event(
                        note_end - start, Event(prog, False, note.pitch)
                    )

        cur_onset = None
        for time in sorted(dict_event.keys()):
            list_events.append(self.voc_dict["time"] + time)
            for event in sorted(dict_event[time], key=attrgetter("pitch", "onset")):
                if cur_onset != event.onset:
                    cur_onset = event.onset
                    list_events.append(self.voc_dict["onset"] + int(event.onset))
                list_events.append(self.voc_dict["note"] + event.pitch)

        # Concatenate tie section, endtie token, and event section
        list_tie_section.append(self.voc_dict["endtie"])
        list_events.append(self.voc_dict["eos"])
        tokens = np.concatenate((list_tie_section, list_events)).astype(int)
        return tokens


def parse_id(voc_dict: dict, id: int):
    keys = voc_dict["keylist"]
    # anchors = [voc_dict[k] for k in keys]
    token_name = keys[0]
    for k in keys:
        if id < voc_dict[k]:
            break
        token_name = k
    token_id = id - voc_dict[token_name]
    return token_name, token_id


def to_second(n: int):
    return n * FRAME_STEP_SIZE_SEC


def find_note(list: list, n: int):
    li_elem = [a for a, _ in list]
    try:
        idx = li_elem.index(n)
    except ValueError:
        return -1
    return idx


def token_seg_list_to_midi(token_seg_list: torch.LongTensor):
    """
    Transform a list of token sequences into a MIDI file.
    """
    midi_data = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    list_onset = []
    cur_time = 0
    for token_seg in token_seg_list:
        list_tie = []
        cur_relative_time = -1
        cur_onset = -1
        tie_end = False
        for token in token_seg:
            token_name, token_id = parse_id(voc_single_track, token)
            if token_name == "note":
                if not tie_end:
                    list_tie.append(token_id)
                elif cur_onset == 1:
                    list_onset.append((token_id, cur_time + cur_relative_time))
                elif cur_onset == 0:
                    i = find_note(list_onset, token_id)
                    if i >= 0:
                        start = list_onset[i][1]
                        end = cur_time + cur_relative_time
                        if start < end:
                            new_note = pm.Note(100, token_id, start, end)
                            piano.notes.append(new_note)
                        list_onset.pop(i)

            elif token_name == "onset":
                if tie_end:
                    if token_id == 1:
                        cur_onset = 1
                    elif token_id == 0:
                        cur_onset = 0
            elif token_name == "time":
                if tie_end:
                    cur_relative_time = to_second(token_id)
            elif token_name == "endtie":
                tie_end = True
                for note, start in list_onset:
                    if note not in list_tie:
                        if start < cur_time:
                            new_note = pm.Note(100, note, start, cur_time)
                            piano.notes.append(new_note)
                        list_onset.remove((note, start))
        cur_time += AUDIO_SEGMENT_SEC

    midi_data.instruments.append(piano)
    return midi_data

import mir_eval
import numpy as np
import pretty_midi as pm


def extract_midi(midi: pm.PrettyMIDI, program=0):
    intervals = []
    pitches = []
    pm_notes = midi.instruments[program].notes
    for note in pm_notes:
        intervals.append((note.start, note.end))
        pitches.append(note.pitch)

    return np.array(intervals), np.array(pitches)


def evaluate_midi(est_midi: pm.PrettyMIDI, ref_midi: pm.PrettyMIDI, program=0):
    est_intervals, est_pitches = extract_midi(est_midi, program)
    ref_intervals, ref_pitches = extract_midi(ref_midi, program)

    dict_eval = mir_eval.transcription.evaluate(
        ref_intervals, ref_pitches, est_intervals, est_pitches
    )

    return dict_eval

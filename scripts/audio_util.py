"""Audio helpers for the multimodal profiler / eval. Decodes to 16 kHz mono float32
via ffmpeg (no python audio libs needed) and indexes LibriSpeech with transcripts."""
import glob
import os
import subprocess
import numpy as np

SR = 16000


def load_wav(path: str, sr: int = SR) -> np.ndarray:
    """Decode any audio file to a mono float32 waveform at ``sr`` Hz via ffmpeg."""
    out = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def librispeech_index(root: str):
    """Return [(flac_path, transcript_text), ...] for a LibriSpeech dir."""
    items = []
    for trans in glob.glob(os.path.join(root, "**", "*.trans.txt"), recursive=True):
        d = os.path.dirname(trans)
        for line in open(trans):
            line = line.strip()
            if not line:
                continue
            uid, _, text = line.partition(" ")
            fp = os.path.join(d, uid + ".flac")
            if os.path.exists(fp):
                items.append((fp, text))
    items.sort()                                  # deterministic
    return items


_STOP = set("the a an and or but of to in on at for with is are was were be been "
            "he she it they we you i his her their our my your that this these those "
            "as by from up out so if then than not no yes do did have has had will "
            "would could should can may might".split())


def content_words(text: str):
    """Lowercase content words (drop stopwords + very short tokens)."""
    import re
    toks = re.findall(r"[a-z']+", text.lower())
    return [t for t in toks if len(t) > 2 and t not in _STOP]


def transcription_overlap(resp: str, transcript: str) -> float:
    """Fraction of the transcript's content words that appear in the response."""
    ref = set(content_words(transcript))
    if not ref:
        return 0.0
    hyp = set(content_words(resp))
    return len(ref & hyp) / len(ref)

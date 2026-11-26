"""Dataset side of the dialect-tag / Standard Greek mixing (cyasr.train_qwen), without loading a model."""
import numpy as np
import soundfile as sf
from torch.utils.data import ConcatDataset

from cyasr.train_qwen import QwenSegmentDataset


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def make_split(root, name, n):
    (root / name / "train").mkdir(parents=True)
    rows = []
    for i in range(n):
        sf.write(root / name / "train" / f"{name}_{i}.wav", np.zeros(1600, dtype="float32"), 16_000)
        rows.append({"id": f"{name}_{i}", "text": f"λέξη {i}"})
    return rows


def test_tag_and_mixing(tmp_path):
    cy_rows, smg_rows = make_split(tmp_path, "cy", 6), make_split(tmp_path, "smg", 3)
    table = {"cy": cy_rows, "smg": smg_rows}
    read_rows = lambda d, split, limit: table[d.name][:limit] if limit else table[d.name]  # noqa: E731
    cy = QwenSegmentDataset(tmp_path / "cy", "train", FakeTokenizer(), None, read_rows, prompt="Cypriot Greek")
    smg = QwenSegmentDataset(tmp_path / "smg", "train", FakeTokenizer(), None, read_rows, prompt=None)
    mixed = ConcatDataset([cy, smg])
    assert len(mixed) == 9
    assert [mixed[i]["prompt"] for i in range(9)] == ["Cypriot Greek"] * 6 + [None] * 3
    assert mixed[0]["audio"].shape == (1600,)


def test_subset_is_evenly_spaced_and_deterministic(tmp_path):
    rows = make_split(tmp_path, "cy", 10)
    read_rows = lambda d, split, limit: rows  # noqa: E731
    ds = QwenSegmentDataset(tmp_path / "cy", "train", FakeTokenizer(), None, read_rows, subset=4)
    assert [r["id"] for r in ds.rows] == ["cy_0", "cy_3", "cy_6", "cy_9"]

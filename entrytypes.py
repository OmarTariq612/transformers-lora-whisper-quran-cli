from dataclasses import dataclass


@dataclass
class WERInfo:
    insertions: int
    deletions: int
    hits: int
    substitutions: int
    wer: float

    def __str__(self) -> str:
        return f"{self.insertions},{self.deletions},{self.hits},{self.substitutions},{self.wer:.4f}"


@dataclass
class PerSorahEntry:
    sorah: int
    wer_info: WERInfo

    def __str__(self) -> str:
        return f"{self.sorah},{self.wer_info}"


@dataclass
class PerAyahEntry:
    sorah: int
    ayah: int
    pred_text: str
    ref_text: str
    wer_info: WERInfo
    duration: float

    def __str__(self) -> str:
        return f"{self.sorah},{self.ayah},{self.pred_text},{self.ref_text},{self.wer_info},{self.duration:.2f}"


@dataclass
class TotalEntry:
    wer_info: WERInfo

    def __str__(self) -> str:
        return f"{self.wer_info}"

from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel
from pyarabic import araby
import csv
import evaluate
import pathlib
from typing import Optional
from utils import path_join, sorah_ayah_format, merge_wer_info
from entrytypes import PerAyahEntry, PerSorahEntry, TotalEntry, WERInfo
from mutagen.mp3 import MP3
from typing import Final
from torch import cuda
import torch
from jiwer import process_words


DEVICE: Final[str] = "cuda" if cuda.is_available() else "cpu"


def transcribe(
    audio_path: str,
    text_csv_path: str,
    peft_model_id: str,
    base_model_id: str,
    device: str = DEVICE,
    from_sorah: int = 1,
    to_sorah: int = 114,
    output_dir: str = ".",
    out_prefix: Optional[str] = None,
    log_level: str = "normal",
    batch_size: int = 8,
):
    if not out_prefix:
        out_prefix = peft_model_id

    reference_texts: dict[int, list[str]] = {}
    with open(text_csv_path, "r") as reference_csv_file:
        reader = csv.DictReader(reference_csv_file)
        for line in reader:
            # TODO: currently we are depending on the order is there a better way??
            reference_texts.setdefault(int(line["sorah"]), list()).append(line["text"])

    audio_dir_path = pathlib.Path(audio_path)
    output_dir_path = pathlib.Path(output_dir)

    for sorah_num in range(from_sorah, to_sorah + 1):  # inclusive
        if not sorah_num in reference_texts:
            raise ValueError(
                f"the given text csv reference doesn't have sorah({sorah_num})"
            )
        if not audio_dir_path.joinpath(
            pathlib.Path(sorah_ayah_format(sorah_num=sorah_num, ayah_num=1))
        ).is_file():
            raise ValueError(f"the given audio path doesn't have sorah({sorah_num})")

    language = "Arabic"
    task = "transcribe"

    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, load_in_8bit=True, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)

    tokenizer = WhisperTokenizer.from_pretrained(
        base_model_id, language=language, task=task
    )
    processor = WhisperProcessor.from_pretrained(
        base_model_id, language=language, task=task
    )
    feature_extractor = processor.feature_extractor
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model, tokenizer=tokenizer, feature_extractor=feature_extractor
    )

    audio_paths = []
    for sorah_num in range(from_sorah, to_sorah + 1):
        for ayah_num in range(1, len(reference_texts[sorah_num]) + 1):
            audio_paths.append(
                path_join(audio_dir_path, sorah_ayah_format(sorah_num, ayah_num))
            )

    with torch.cuda.amp.autocast():
        outputs = pipe(
            audio_paths,
            chunk_length_s=30,
            batch_size=batch_size,
            generate_kwargs={
                "task": "transcribe",
                "language": "ar",
                # "forced_decoder_ids": forced_decoder_ids,
                "max_new_tokens": 255,
            },
        )

    wer_module = evaluate.load("wer")
    per_sorah: list[PerSorahEntry] = []
    per_ayah: list[PerAyahEntry] = []
    per_ayah_index = 0

    for sorah_num in range(from_sorah, to_sorah + 1):
        for ayah_num in range(1, len(reference_texts[sorah_num]) + 1):
            index = per_ayah_index + ayah_num - 1
            prediction_text = araby.strip_diacritics(outputs[index]["text"].strip())
            if log_level == "verbose":
                print(prediction_text)

            ayah_ref_text = reference_texts[sorah_num][ayah_num - 1]
            wer: float = wer_module.compute(
                predictions=[prediction_text], references=[ayah_ref_text]
            )
            word_output = process_words(
                hypothesis=prediction_text, reference=ayah_ref_text
            )
            per_ayah.append(
                PerAyahEntry(
                    sorah=sorah_num,
                    ayah=ayah_num,
                    pred_text=prediction_text,
                    ref_text=ayah_ref_text,
                    wer_info=WERInfo(
                        insertions=word_output.insertions,
                        deletions=word_output.deletions,
                        hits=word_output.hits,
                        substitutions=word_output.substitutions,
                        wer=wer,
                    ),
                    duration=MP3(
                        path_join(
                            audio_dir_path, sorah_ayah_format(sorah_num, ayah_num)
                        )
                    ).info.length,  # type: ignore
                )
            )

        per_sorah.append(
            PerSorahEntry(
                sorah=sorah_num,
                wer_info=merge_wer_info(
                    map(lambda entry: entry.wer_info, per_ayah[per_ayah_index:])
                ),
            )
        )

        per_ayah_index = len(per_ayah)

    with open(
        path_join(output_dir_path, f"{out_prefix}_per_ayah.csv"), "w", encoding="utf-8"
    ) as per_ayah_file:
        per_ayah_file.write(
            "sorah,ayah,pred_text,ref_text,insertions,deletions,hits,substitutions,wer,duration\n"
        )
        for entry in per_ayah:
            per_ayah_file.write(f"{entry}\n")

    with open(
        path_join(output_dir_path, f"{out_prefix}_per_sorah.csv"), "w"
    ) as per_sorah_file:
        per_sorah_file.write("sorah,insertions,deletions,hits,substitutions,wer\n")
        for entry in per_sorah:  # type: ignore
            per_sorah_file.write(f"{entry}\n")

    total_entry = TotalEntry(
        wer_info=merge_wer_info(map(lambda entry: entry.wer_info, per_sorah))
    )

    with open(path_join(output_dir_path, f"{out_prefix}_total.csv"), "w") as total_file:
        total_file.write("insertions,deletions,hits,substitutions,wer\n")
        total_file.write(f"{total_entry}\n")

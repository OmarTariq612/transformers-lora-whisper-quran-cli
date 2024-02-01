# transformers-lora-whisper-quran-cli

## Requirements:

```bash
pip3 install -U click \
  mutagen \
  jiwer \
  evaluate \
  transformers \
  pyarabic \
  optimum \
  "git+https://github.com/huggingface/peft.git@main" \
  bitsandbytes
```

```bash
python3 main.py --help
```

## Examples

* Calculate WER and generate csv files for juz 28 `(58:66)` using custom checkpoint from a huggingface repo.

```bash
python3 main.py generate \
  --sorah-range 58:66 \
  "/kaggle/input/quran-reciters/metadata/ayah_text_clear.csv" \
  "/kaggle/input/quran-reciters/audio/audio/Minshawy_Murattal_128kbps" \
  --peft-model "omartariq612/quran-whisper-large-v3-epoch-5" \
  --base-model "openai/whisper-large-v3"  \
  --out-prefix "Minshawy_Murattal_128kbps" \
  -o .. \
  --device cuda \
  --batch-size 24
```
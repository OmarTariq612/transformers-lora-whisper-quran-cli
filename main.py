import click
from clicktypes import SORAH_RANGE
from merge import merge as m


@click.group()
def main():
    pass


@main.command(help="generate csv files containing WER for the given input and model")
@click.argument(
    "text-csv-path", type=click.Path(dir_okay=False, exists=True, readable=True)
)
@click.argument(
    "audio-path", type=click.Path(file_okay=False, exists=True, executable=True)
)
@click.option(
    "--peft-model",
    type=str,
    help="multilingual peft model used for transcribing",
)
@click.option(
    "--base-model",
    type=str,
    help="the base-model of the given peft model",
)
@click.option("--device", type=str, default="cpu", help="device used to load the model")
@click.option("--sorah-range", default="1:114", type=SORAH_RANGE)
@click.option(
    "--out-prefix",
    type=click.STRING,
)
@click.option(
    "--log-level",
    default="normal",
    type=click.Choice(("normal", "verbose")),
    help="determine the logging level (default: normal)",
)
@click.option(
    "-o",
    default=".",
    type=click.Path(file_okay=False, executable=True),
    help="output directory",
)
@click.option(
    "--batch-size",
    "-b",
    default=8,
    type=click.INT,
    help="batch size used for transcribing (default: 8)",
)
def generate(
    text_csv_path: str,
    audio_path: str,
    peft_model: str,
    base_model: str,
    device: str,
    sorah_range: tuple[int, int],
    out_prefix: str,
    log_level: str,
    o: str,
    batch_size: int,
):
    from transcribe import transcribe

    transcribe(
        audio_path=audio_path,
        text_csv_path=text_csv_path,
        peft_model_id=peft_model,
        base_model_id=base_model,
        device=device,
        from_sorah=sorah_range[0],
        to_sorah=sorah_range[1],
        output_dir=o,
        out_prefix=out_prefix,
        log_level=log_level,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()

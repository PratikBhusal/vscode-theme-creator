from pathlib import Path
from typing import Annotated

import typer

from colorscheme import COLORS_DIR, THEMES_DIR

app = typer.Typer(invoke_without_command=True)


@app.callback()  # type: ignore[misc]
def default(ctx: typer.Context) -> None:
    """Run extract then restore when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        extract()
        restore()


INPUT_EXTRACT_DIR: Path = Path("input-extract")
INPUT_RESTORE_DIR: Path = Path("input-restore")


@app.command()  # type: ignore[misc]
def extract(
    oklch: Annotated[
        bool,
        typer.Option(
            "--oklch",
            help="Convert hex values to oklch(). Mutually exclusive with --hex.",
        ),
    ] = False,
    hex_: Annotated[
        bool,
        typer.Option(
            "--hex",
            help="Convert oklch() values to hex. Mutually exclusive with --oklch.",
        ),
    ] = False,
) -> None:
    """Extract color values from all theme JSON files in input-extract/ into color map JSON files."""
    from colorscheme import extract as _extract

    input_files: list[Path] = sorted(INPUT_EXTRACT_DIR.glob("*.json"))
    if not input_files:
        typer.echo(f"No JSON files found in {INPUT_EXTRACT_DIR}/", err=True)
        raise typer.Exit(1)

    _extract(input_files=input_files, output_dir=COLORS_DIR, oklch=oklch, hex_=hex_)


@app.command()  # type: ignore[misc]
def restore(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o", help="Directory to write restored themes into."
        ),
    ] = THEMES_DIR,
    oklch: Annotated[
        bool,
        typer.Option(
            "--oklch",
            help="Convert hex values to oklch(). Mutually exclusive with --hex.",
        ),
    ] = False,
    hex_: Annotated[
        bool,
        typer.Option(
            "--hex",
            help="Convert oklch() values to hex. Mutually exclusive with --oklch.",
        ),
    ] = False,
) -> None:
    """Restore theme JSON files from color map JSON files in input-restore/."""
    from colorscheme import restore as _restore

    colors_files: list[Path] = sorted(INPUT_RESTORE_DIR.glob("*-colors.json"))
    if not colors_files:
        typer.echo(f"No color map files found in {INPUT_RESTORE_DIR}/", err=True)
        raise typer.Exit(1)

    for colors_file in colors_files:
        _restore(
            colors_file=colors_file,
            theme_file=None,
            output_file=None,
            output_dir=output_dir,
            oklch=oklch,
            hex_=hex_,
        )


@app.command()  # type: ignore[misc]
def collate(
    output_file: Annotated[
        Path,
        typer.Option(
            "--output-file", "-o", help="Path to write the collated JSON file."
        ),
    ] = Path("collated.json"),
) -> None:
    """Collate all color map files in colors/ into a single key-to-colorschemes index."""
    from colorscheme import collate as _collate

    colors_files: list[Path] = sorted(COLORS_DIR.glob("*-colors.json"))
    if not colors_files:
        typer.echo(f"No color map files found in {COLORS_DIR}/", err=True)
        raise typer.Exit(1)

    _collate(input_files=colors_files, output_file=output_file)


if __name__ == "__main__":
    app()

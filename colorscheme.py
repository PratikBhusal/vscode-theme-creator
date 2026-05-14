import json
import re
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Annotated, TypedDict, cast

import numpy as np
import numpy.typing as npt
import typer

warnings.filterwarnings("ignore", module="colour.*")
import colour  # noqa: E402

app: typer.Typer = typer.Typer()


class _ThemeBaseRequired(TypedDict):
    name: str


class _SemanticTokenStyle(TypedDict, total=False):
    foreground: str
    fontStyle: str
    bold: bool
    italic: bool
    underline: bool
    strikethrough: bool


class _ThemeBase(_ThemeBaseRequired, total=False):
    type: str
    author: str
    uuid: str
    colorSpaceName: str
    include: str
    semanticHighlighting: bool
    semanticTokenColors: dict[str, str | _SemanticTokenStyle]


class _TokenSettings(TypedDict, total=False):
    foreground: str
    background: str
    fontStyle: str
    content: str


class _TokenColorEntryRequired(TypedDict):
    settings: _TokenSettings


class _TokenColorEntry(_TokenColorEntryRequired, total=False):
    name: str
    scope: str | list[str]


class _ThemeJson(_ThemeBase, total=False):
    colors: dict[str, str]
    tokenColors: list[_TokenColorEntry]


class _MetaJson(_ThemeBase, total=False):
    _key_order: list[str]
    _token_metadata: dict[str, dict[str, str]]
    _semantic_token_metadata: dict[str, _SemanticTokenStyle]


class _TokenEntry(TypedDict):
    name: str
    scopes: set[str]
    settings: dict[str, str]  # prop -> color value


COLORS_DIR: Path = Path("colors")
THEMES_DIR: Path = Path("themes")

# Parses oklch(L C H) or oklch(L C H / alpha) into capture groups
_OKLCH_RE: re.Pattern[str] = re.compile(
    r"oklch\(([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*/\s*([\d.]+))?\)"
)


def _hex_to_oklch(hex_color: str) -> str:
    """Convert a hex color string to an oklch() CSS function string, preserving alpha."""
    h: str = hex_color.lstrip("#")
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    rgb: npt.NDArray[np.float64] = (
        np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)]) / 255.0  # type: ignore[misc]
    )
    oklch: npt.NDArray[np.float64] = colour.Oklab_to_Oklch(  # type: ignore[attr-defined]
        colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(rgb))
    )
    lightness: float
    chroma: float
    hue_angle: float
    lightness, chroma, hue_angle = oklch
    if len(h) == 8:
        alpha: float = int(h[6:8], 16) / 255.0
        return f"oklch({lightness:.4f} {chroma:.4f} {hue_angle:.2f} / {alpha:.4f})"
    return f"oklch({lightness:.4f} {chroma:.4f} {hue_angle:.2f})"


def _oklch_to_hex(oklch_str: str) -> str:
    """Convert an oklch() CSS function string back to a hex color, restoring alpha if present."""
    m: re.Match[str] | None = _OKLCH_RE.match(oklch_str)
    if not m:
        raise ValueError(f"Invalid oklch string: {oklch_str}")
    lightness: float
    chroma: float
    hue_angle: float
    lightness, chroma, hue_angle = float(m[1]), float(m[2]), float(m[3])
    oklch_arr: npt.NDArray[np.float64] = np.array([lightness, chroma, hue_angle])  # type: ignore[misc]
    oklab: npt.NDArray[np.float64] = colour.Oklch_to_Oklab(oklch_arr)  # type: ignore[attr-defined]
    rgb: npt.NDArray[np.float64] = np.clip(
        colour.XYZ_to_sRGB(colour.Oklab_to_XYZ(oklab)), 0, 1
    )
    hex_color: str = "#{:02x}{:02x}{:02x}".format(
        *(int(round(float(v) * 255)) for v in rgb)  # type: ignore[misc]
    )
    if m[4] is not None:
        hex_color += f"{int(round(float(m[4]) * 255)):02x}"
    return hex_color


def _convert_color(color: str, *, to_hex: bool, to_oklch: bool) -> str:
    """Convert a color between hex and oklch formats. Normalizes oklch values through hex for consistent precision."""
    if not color:
        return color
    is_hex: bool = color.startswith("#")
    if to_oklch:
        return _hex_to_oklch(color if is_hex else _oklch_to_hex(color))
    if to_hex and not is_hex:
        return _oklch_to_hex(color)
    return color


def _invert_colors(
    color_map: dict[str, list[str]],
    to_hex: bool = False,
    to_oklch: bool = False,
) -> dict[str, str]:
    """Expand {color: [key, ...]} back to {key: color}, optionally converting format."""
    return {
        key: _convert_color(color, to_hex=to_hex, to_oklch=to_oklch)
        for color, keys in color_map.items()
        for key in keys
    }


def _collate(input_files: list[Path]) -> dict[str, list[str]]:
    result: defaultdict[str, list[str]] = defaultdict(list)
    for colors_file in input_files:
        data: dict[str, object] = json.loads(colors_file.read_text())
        meta: dict[str, object] = cast(dict[str, object], data.get("meta", {}))
        name: str = cast(str, meta.get("name", colors_file.stem.removesuffix("-colors")))
        for refs in cast(dict[str, list[str]], data["colors"]).values():
            for ref in refs:
                result[ref].append(name)
    return dict(sorted(result.items()))


def _extract(
    src: Path, dst: Path, to_hex: bool = False, to_oklch: bool = False
) -> None:
    with open(src) as f:
        raw: str = f.read()
    # Captures the contents of the "colors" block via regex (before full JSON parsing)
    colors_block: re.Match[str] | None = re.search(
        r'"colors"\s*:\s*\{(.+?)\n\s*\}', raw, re.DOTALL
    )
    convert: partial[str] = partial(_convert_color, to_hex=to_hex, to_oklch=to_oklch)

    # Matches JSON "key": "#hex" pairs (3/4/6/8-digit hex colors)
    hex_re: re.Pattern[str] = re.compile(
        r'"([^"]+)"\s*:\s*"(#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3}))"'
    )
    # Matches JSON "key": "oklch(...)" pairs
    oklch_val_re: re.Pattern[str] = re.compile(
        r'"([^"]+)"\s*:\s*"(oklch\([\d.\s/]+\))"'
    )
    # Matches JSON "key": "" pairs (empty string values)
    empty_val_re: re.Pattern[str] = re.compile(r'"([^"]+)"\s*:\s*""')

    mapping: defaultdict[str, list[str]] = defaultdict(list)
    if colors_block is not None:
        block: str = colors_block.group(1)
        for m in hex_re.finditer(block):
            mapping[convert(m[2])].append(m[1])
        for m in oklch_val_re.finditer(block):
            mapping[convert(m[2])].append(m[1])
        for m in empty_val_re.finditer(block):
            mapping[""].append(m[1])

    theme: _ThemeJson = json.loads(raw)
    meta: _MetaJson = cast(
        _MetaJson,
        {
            k: v
            for k, v in theme.items()
            if k not in ("colors", "tokenColors", "semanticTokenColors")
        },
    )
    # Store original top-level key order so restore can reproduce it exactly
    meta["_key_order"] = list(theme.keys())

    # Extract colors from tokenColors[i].settings.{foreground,background}
    token_colors_raw: list[_TokenColorEntry] = theme.get("tokenColors", [])
    # Per-token metadata for round-trip fidelity (scope type, extra settings)
    token_metadata: dict[str, dict[str, str]] = {}
    for i, token in enumerate(token_colors_raw):
        settings: _TokenSettings = token.get("settings", {})
        name: str = token.get("name", "")
        # Entries that omit "scope" set a global token default; sentinel preserves this
        if "scope" not in token:
            scopes: list[str] = ["global_scope"]
        else:
            scope: str | list[str] = token["scope"]
            scopes = scope if isinstance(scope, list) else [scope]
        # All non-color settings (fontStyle, content, etc.) preserved in metadata
        extra_settings: dict[str, str] = {
            k: cast(str, v)
            for k, v in settings.items()
            if k not in ("foreground", "background")
            and not (k == "fontStyle" and v == "")
        }
        if extra_settings:
            token_metadata[str(i)] = extra_settings
        has_color: bool = False
        for prop, color_val in (
            ("foreground", settings.get("foreground")),
            ("background", settings.get("background")),
        ):
            if not color_val:
                continue
            is_hex: bool = color_val.startswith("#")
            if is_hex or color_val.startswith("oklch("):
                has_color = True
                converted: str = convert(color_val)
                for scope_str in scopes:
                    mapping[converted].append(
                        f"tokenColors[{name}][{scope_str}][{prop}][{i}]"
                    )
        # Colorless entries: emit sentinel ref so they're not lost
        if not has_color:
            for scope_str in scopes:
                mapping[""].append(f"tokenColors[{name}][{scope_str}][{i}]")

    if token_metadata:
        meta["_token_metadata"] = token_metadata

    semantic_token_colors: dict[str, str | _SemanticTokenStyle] = theme.get(
        "semanticTokenColors", {}
    )
    semantic_meta: dict[str, _SemanticTokenStyle] = {}
    for token_key, val in semantic_token_colors.items():
        if isinstance(val, str):
            if val.startswith("#") or val.startswith("oklch("):
                mapping[convert(val)].append(f"semanticTokenColors[{token_key}]")
            else:
                mapping[""].append(f"semanticTokenColors[{token_key}]")
        elif isinstance(val, dict):
            has_fg: bool = False
            non_color: dict[str, str | bool] = {}
            for prop, prop_val in val.items():
                if (
                    prop == "foreground"
                    and isinstance(prop_val, str)
                    and (prop_val.startswith("#") or prop_val.startswith("oklch("))
                ):
                    has_fg = True
                    mapping[convert(prop_val)].append(
                        f"semanticTokenColors[{token_key}][foreground]"
                    )
                else:
                    if not (prop == "fontStyle" and prop_val == ""):
                        non_color[prop] = cast(str | bool, prop_val)
            if non_color:
                semantic_meta[token_key] = cast(_SemanticTokenStyle, non_color)
            if not has_fg:
                mapping[""].append(f"semanticTokenColors[{token_key}]")
    if semantic_meta:
        meta["_semantic_token_metadata"] = semantic_meta

    color_map: dict[str, list[str]] = dict(
        sorted(
            ((color, sorted(refs)) for color, refs in mapping.items()),
            key=lambda x: x[1][0],  # type: ignore[misc]
        )
    )
    out: dict[str, object] = {"colors": color_map, "meta": meta}
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    typer.echo(f"{dst}: {len(color_map)} colors")


@app.command()  # type: ignore[misc]
def extract(
    input_files: Annotated[
        list[Path],
        typer.Argument(help="Theme JSON files to process."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o", help="Directory to write color map files into."
        ),
    ] = COLORS_DIR,
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
    """Extract color values from VSCode theme JSON files into color map JSON files.

    By default, color values are passed through unchanged. Use --oklch or --hex
    to convert between formats.
    """
    if oklch and hex_:
        raise typer.BadParameter("--oklch and --hex are mutually exclusive.")
    output_dir.mkdir(exist_ok=True)
    for src in input_files:
        dst: Path = output_dir / (src.stem + "-colors.json")
        _extract(src, dst, to_hex=hex_, to_oklch=oklch)


@app.command()  # type: ignore[misc]
def restore(
    colors_file: Annotated[Path, typer.Argument(help="Color map JSON file.")],
    theme_file: Annotated[
        Path | None,
        typer.Argument(
            help="Source theme JSON file. Uses metadata embedded in the colors file if omitted."
        ),
    ] = None,
    output_file: Annotated[Path | None, typer.Argument(help="Output path.")] = None,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Directory to write the restored theme into. Ignored if output_file is given.",
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
    """Restore a theme JSON file from a color map JSON file.

    By default, color values are passed through unchanged. Use --oklch or --hex
    to convert between formats. If the colors file contains embedded metadata
    (written by extract), the theme file argument may be omitted.
    """
    if oklch and hex_:
        raise typer.BadParameter("--oklch and --hex are mutually exclusive.")

    with open(colors_file) as f:
        data: dict[str, object] = json.load(f)

    # Support both the new {meta, colors} format and bare {color: [keys]} format
    if "colors" in data and "meta" in data:
        color_map: dict[str, list[str]] = cast(dict[str, list[str]], data["colors"])
        base_theme: dict[str, object] = dict(cast(dict[str, object], data["meta"]))
    else:
        color_map = cast(dict[str, list[str]], data)
        base_theme = {}

    if theme_file is not None:
        with open(theme_file) as f:
            base_theme = json.load(f)

    if not base_theme:
        raise typer.BadParameter(
            "No theme metadata found. Pass a theme file or use a colors file created by extract."
        )

    theme_name: str = (
        theme_file.name
        if theme_file
        else (colors_file.stem.removesuffix("-colors") + ".json")
    )
    out_path: Path = output_file if output_file is not None else output_dir / theme_name
    output_dir.mkdir(exist_ok=True)

    inverted: dict[str, str] = _invert_colors(color_map, to_hex=hex_, to_oklch=oklch)

    # Matches tokenColors[name][scope][foreground|background][index] keys
    token_color_re: re.Pattern[str] = re.compile(
        r"^tokenColors\[([^\]]*)\]\[([^\]]*)\]\[(foreground|background)\]\[(\d+)\]$"
    )
    # Matches colorless tokenColors[name][scope][index] sentinel keys
    token_sentinel_re: re.Pattern[str] = re.compile(
        r"^tokenColors\[([^\]]*)\]\[([^\]]*)\]\[(\d+)\]$"
    )
    # Matches semanticTokenColors[token_key][foreground] — dict-type entry
    semantic_fg_re: re.Pattern[str] = re.compile(
        r"^semanticTokenColors\[([^\]]*)\]\[foreground\]$"
    )
    # Matches semanticTokenColors[token_key] — direct string color or sentinel
    semantic_str_re: re.Pattern[str] = re.compile(r"^semanticTokenColors\[([^\]]*)\]$")

    theme_colors: dict[str, str] = {}
    token_entries: dict[int, _TokenEntry] = {}
    semantic_dict_entries: dict[str, dict[str, str]] = {}
    semantic_str_entries: dict[str, str] = {}
    semantic_sentinel_keys: set[str] = set()
    for key, color in inverted.items():
        if color == "":
            ms: re.Match[str] | None = token_sentinel_re.match(key)
            sem_s: re.Match[str] | None = semantic_str_re.match(key)
            if ms:
                idx: int = int(ms[3])
                entry: _TokenEntry = token_entries.setdefault(
                    idx, {"name": ms[1], "scopes": set(), "settings": {}}
                )
                entry["scopes"].add(ms[2])
            elif sem_s:
                semantic_sentinel_keys.add(sem_s[1])
            else:
                # Plain color key with an empty string
                # value (e.g. diffEditor.move.border)
                theme_colors[key] = color
            continue
        m: re.Match[str] | None = token_color_re.match(key)
        sem_fg: re.Match[str] | None = semantic_fg_re.match(key)
        sem_s = semantic_str_re.match(key)
        if m:
            idx = int(m[4])
            entry = token_entries.setdefault(
                idx, {"name": m[1], "scopes": set(), "settings": {}}
            )
            entry["scopes"].add(m[2])
            entry["settings"].setdefault(m[3], color)
        elif sem_fg:
            semantic_dict_entries[sem_fg[1]] = {"foreground": color}
        elif sem_s:
            semantic_str_entries[sem_s[1]] = color
        else:
            theme_colors[key] = color

    key_order: list[str] = cast(list[str], base_theme.pop("_key_order", []))
    token_metadata_by_idx: dict[str, dict[str, str]] = cast(
        dict[str, dict[str, str]], base_theme.pop("_token_metadata", {})
    )
    semantic_token_metadata: dict[str, _SemanticTokenStyle] = cast(
        dict[str, _SemanticTokenStyle], base_theme.pop("_semantic_token_metadata", {})
    )

    if not key_order or "colors" in key_order or theme_colors:
        existing_colors: dict[str, str] = cast(
            dict[str, str], base_theme.get("colors", {})
        )
        merged_colors: dict[str, str] = {**existing_colors, **theme_colors}
        base_theme["colors"] = dict(sorted(merged_colors.items()))

    token_colors: list[_TokenColorEntry] = []
    for idx in sorted(token_entries):
        entry = token_entries[idx]
        scopes: list[str] = sorted(entry["scopes"])
        tmeta: dict[str, str] = token_metadata_by_idx.get(str(idx), {})
        is_global: bool = scopes == ["global_scope"]
        scope_val: str | list[str] = scopes if len(scopes) > 1 else scopes[0]
        settings: _TokenSettings = cast(
            _TokenSettings,
            dict(sorted({**entry["settings"], **tmeta}.items())),  # type: ignore[misc]
        )
        token_name: str = entry["name"]
        token_entry: _TokenColorEntry = {"settings": settings}
        if token_name:
            token_entry["name"] = token_name
        if not is_global:
            token_entry["scope"] = scope_val
        token_colors.append(token_entry)
    if not key_order or "tokenColors" in key_order or token_colors:
        base_theme["tokenColors"] = token_colors

    semantic_colors: dict[str, str | dict[str, str | bool]] = {}
    for token_key, fg_entry in semantic_dict_entries.items():
        extra: _SemanticTokenStyle = semantic_token_metadata.get(token_key, {})
        semantic_colors[token_key] = {**fg_entry, **cast(dict[str, str | bool], extra)}
    for token_key in semantic_sentinel_keys:
        sem_meta: _SemanticTokenStyle | None = semantic_token_metadata.get(token_key)
        if sem_meta is not None:
            semantic_colors[token_key] = cast(dict[str, str | bool], dict(sem_meta))
        else:
            semantic_colors[token_key] = ""
    for token_key, color_val in semantic_str_entries.items():
        semantic_colors[token_key] = color_val
    if semantic_colors:
        base_theme["semanticTokenColors"] = semantic_colors

    if key_order:
        ordered_theme: dict[str, object] = {
            k: base_theme[k] for k in key_order if k in base_theme
        }
        ordered_theme.update(
            {k: v for k, v in base_theme.items() if k not in ordered_theme}
        )
        output_theme: dict[str, object] = ordered_theme
    else:
        output_theme = base_theme

    with open(out_path, "w") as f:
        json.dump(output_theme, f, indent=2)
        f.write("\n")
    typer.echo(f"Written to {out_path}")


@app.command()  # type: ignore[misc]
def collate(
    input_files: Annotated[
        list[Path],
        typer.Argument(help="Color map JSON files to collate."),
    ],
    output_file: Annotated[
        Path,
        typer.Option("--output-file", "-o", help="Path to write the collated JSON file."),
    ] = Path("collated.json"),
) -> None:
    """Collate color map files into a single index mapping each VSCode key to the colorschemes that define it."""
    result: dict[str, list[str]] = _collate(input_files)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    typer.echo(f"{output_file}: {len(result)} keys")


if __name__ == "__main__":
    app()

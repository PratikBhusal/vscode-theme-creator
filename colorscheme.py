import json
import re
import shutil
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
OTHERS_DIR: Path = Path("others")

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


_TOKEN_IDX_RE: re.Pattern[str] = re.compile(r"\[\d+\]$")
_EXPAND_SEMANTIC_PROP_RE: re.Pattern[str] = re.compile(
    r"^\[([^\]]+)\]\[([^\]]+)\]$"
)
_EXPAND_SEMANTIC_DIRECT_RE: re.Pattern[str] = re.compile(
    r"^\[([^\]]+)\]$"
)
_EXPAND_TOKEN_RE: re.Pattern[str] = re.compile(
    r"^\[(.*)\]\[([^\]]+)\]\[([^\]]+)\]$"
)
# Colorless sentinel refs (no foreground/background prop bracket)
_EXPAND_TOKEN_COLORLESS_RE: re.Pattern[str] = re.compile(
    r"^\[(.*)\]\[([^\]]+)\]$"
)


def _collate(input_files: list[Path]) -> dict[str, object]:
    result: defaultdict[str, list[str]] = defaultdict(list)
    top_level: defaultdict[str, list[str]] = defaultdict(list)
    for colors_file in input_files:
        data: dict[str, object] = json.loads(colors_file.read_text())
        meta: dict[str, object] = cast(dict[str, object], data.get("meta", {}))
        name: str = cast(
            str, meta.get("name", colors_file.stem.removesuffix("-colors"))
        )
        for refs in cast(dict[str, list[str]], data["colors"]).values():
            for ref in refs:
                key: str = (
                    _TOKEN_IDX_RE.sub("", ref)
                    if ref.startswith("tokenColors")
                    else ref
                )
                result[key].append(f"{name} ({colors_file.stem})")
        for top_key in cast(list[str], meta.get("_key_order", [])):
            top_level[top_key].append(f"{name} ({colors_file.stem})")
    flat: dict[str, list[str]] = {}
    semantic: dict[str, list[str]] = {}
    token: dict[str, list[str]] = {}
    for k, v in sorted(result.items()):
        if k.startswith("semanticTokenColors"):
            semantic[k[len("semanticTokenColors"):]] = v
        elif k.startswith("tokenColors"):
            token[k[len("tokenColors"):]] = v
        else:
            flat[k] = v

    return {
        **flat,
        "semanticTokenColors": semantic,
        "tokenColors": token,
        "topLevelKeys": dict(sorted(top_level.items())),
    }


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
        typer.Option(
            "--output-file", "-o", help="Path to write the collated JSON file."
        ),
    ] = Path("collated.json"),
) -> None:
    """Collate color map files into a single index mapping each VSCode key to the colorschemes that define it."""
    result: dict[str, object] = _collate(input_files)
    n_flat: int = sum(1 for k in result if k not in ("semanticTokenColors", "tokenColors", "topLevelKeys"))
    n_semantic: int = len(cast(dict[str, object], result.get("semanticTokenColors", {})))
    n_token: int = len(cast(dict[str, object], result.get("tokenColors", {})))
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    typer.echo(f"{output_file}: {n_flat} color, {n_semantic} semanticTokenColors, {n_token} tokenColors")


class _PkgTheme(TypedDict, total=False):
    label: str
    uiTheme: str
    path: str


class _PkgContributes(TypedDict, total=False):
    themes: list[_PkgTheme]


class _PkgJson(TypedDict, total=False):
    name: str
    contributes: _PkgContributes


def _load_pkg(pkg_path: Path) -> tuple[str, list[_PkgTheme]]:
    with open(pkg_path) as f:
        data: _PkgJson = json.load(f)
    name: str = data.get("name", "")
    contributes: _PkgContributes = data.get("contributes", _PkgContributes())
    themes: list[_PkgTheme] = contributes.get("themes", [])
    return name, themes


def _theme_paths_from_pkg(pkg_path: Path) -> list[str]:
    _, themes = _load_pkg(pkg_path)
    return [t["path"] for t in themes if "path" in t]


def _check_subdir(subdir: Path) -> tuple[list[str], list[str]]:
    """Return (missing_paths, warnings)."""
    pkg_path = subdir / "package.json"
    if not pkg_path.exists():
        return [], [f"{subdir.name}/ has no package.json"]

    rel_paths = _theme_paths_from_pkg(pkg_path)
    missing: list[str] = [
        f"{subdir.name}/{rel}"
        for rel in rel_paths
        if not (subdir / rel).resolve().exists()
    ]

    count = len(rel_paths)
    status = (
        f"MISSING {len(missing)}/{count}"
        if missing
        else f"OK ({count} theme{'s' if count != 1 else ''})"
    )
    label = f"Checking {subdir.name}..."
    typer.echo(f"{label:<70} {status}")

    return missing, []


@app.command()  # type: ignore[misc]
def check(
    others_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing VSCode extension subdirectories."),
    ] = OTHERS_DIR,
) -> None:
    """Check that every theme path declared in each package.json exists on disk."""
    subdirs: list[Path] = sorted(p for p in others_dir.iterdir() if p.is_dir())

    all_missing: list[str] = []
    all_warnings: list[str] = []
    total_checked: int = 0

    for subdir in subdirs:
        missing, warnings_ = _check_subdir(subdir)
        for w in warnings_:
            typer.echo(f"WARNING: {w}")
        all_missing.extend(missing)
        all_warnings.extend(warnings_)
        pkg_path = subdir / "package.json"
        if pkg_path.exists():
            total_checked += len(_theme_paths_from_pkg(pkg_path))

    typer.echo("")
    if all_missing:
        typer.echo("--- Missing theme files ---")
        for m in all_missing:
            typer.echo(f"  {m}")
        typer.echo("")

    if not all_missing and not all_warnings:
        typer.echo("All theme files present.")
    else:
        parts: list[str] = [f"{total_checked} checked", f"{len(all_missing)} missing"]
        if all_warnings:
            parts.append(
                f"{len(all_warnings)} warning{'s' if len(all_warnings) != 1 else ''}"
            )
        typer.echo(f"Total: {', '.join(parts)}")


@app.command()  # type: ignore[misc]
def collect(
    others_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing VSCode extension subdirectories."),
    ] = OTHERS_DIR,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o", help="Directory to write collected theme files into."
        ),
    ] = THEMES_DIR,
) -> None:
    """Copy theme JSON files from each extension into a flat directory, prefixed with the package name."""
    output_dir.mkdir(exist_ok=True)
    subdirs: list[Path] = sorted(p for p in others_dir.iterdir() if p.is_dir())
    total: int = 0

    for subdir in subdirs:
        pkg_path = subdir / "package.json"
        if not pkg_path.exists():
            typer.echo(f"WARNING: {subdir.name}/ has no package.json")
            continue

        name, themes = _load_pkg(pkg_path)
        for theme in themes:
            rel: str = theme.get("path", "")
            if not rel:
                continue
            src: Path = (subdir / rel).resolve()
            if not src.exists():
                typer.echo(f"WARNING: missing {subdir.name}/{rel}")
                continue
            dst: Path = output_dir / f"{name}-{src.name.replace(' ', '-')}"
            shutil.copy2(src, dst)
            total += 1

    typer.echo(f"Collected {total} theme files into {output_dir}/")


@app.command()  # type: ignore[misc]
def expand(
    colorscheme_file: Annotated[
        Path,
        typer.Argument(help="Input colorscheme file (input-extract format)."),
    ],
    collated_file: Annotated[
        Path,
        typer.Option("--collated", "-c", help="Collated keys catalog."),
    ] = Path("collated.json"),
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Directory to write the expanded theme into."),
    ] = THEMES_DIR,
    placeholder: Annotated[
        str,
        typer.Option("--placeholder", "-p", help="Value inserted for keys absent from the input colorscheme."),
    ] = "TODO",
) -> None:
    """Expand a colorscheme to cover every key in the collated catalog, filling gaps with a placeholder."""
    with open(collated_file) as f:
        collated_raw: dict[str, object] = json.load(f)
    collated_semantic: dict[str, object] = cast(dict[str, object], collated_raw.get("semanticTokenColors", {}))
    collated_token: dict[str, object] = cast(dict[str, object], collated_raw.get("tokenColors", {}))
    flat_keys: list[str] = [k for k in collated_raw if k not in ("semanticTokenColors", "tokenColors", "topLevelKeys")]

    with open(colorscheme_file) as f:
        data: _ThemeJson = json.load(f)

    # semanticTokenColors keys
    semantic_keys: dict[str, set[str]] = {}
    for key in collated_semantic:
        if m := _EXPAND_SEMANTIC_PROP_RE.match(key):
            semantic_keys.setdefault(m.group(1), set()).add("prop")
        elif m := _EXPAND_SEMANTIC_DIRECT_RE.match(key):
            semantic_keys.setdefault(m.group(1), set()).add("direct")

    # tokenColors keys
    # name -> scope -> set of props (foreground, background, …)
    token_keys: dict[str, dict[str, set[str]]] = {}
    for key in collated_token:
        if m := _EXPAND_TOKEN_RE.match(key):
            token_keys.setdefault(m.group(1), {}).setdefault(m.group(2), set()).add(m.group(3))
        elif m := _EXPAND_TOKEN_COLORLESS_RE.match(key):
            token_keys.setdefault(m.group(1), {}).setdefault(m.group(2), set())

    # colors — flat VSCode UI keys only
    source_colors: dict[str, str] = data.get("colors", {})
    out_colors: dict[str, str] = {k: source_colors.get(k, placeholder) for k in flat_keys}
    filled: int = sum(1 for v in out_colors.values() if v != placeholder)
    todo: int = len(out_colors) - filled

    # semanticTokenColors
    input_semantic: dict[str, str | _SemanticTokenStyle] = data.get("semanticTokenColors", {})
    filled_semantic: dict[str, object] = {}
    # List preserves insertion order and allows the same token to appear twice (direct + prop).
    todo_semantic: list[tuple[str, object]] = []
    for tok, kinds in semantic_keys.items():
        if tok in input_semantic:
            filled_semantic[tok] = input_semantic[tok]
        else:
            if "direct" in kinds:
                todo_semantic.append((tok, placeholder))
            if "prop" in kinds:
                todo_semantic.append((tok, {"foreground": placeholder}))

    # tokenColors — build name→settings lookup from input, then reconstruct entries
    input_tokens: list[_TokenColorEntry] = data.get("tokenColors", [])
    token_settings: dict[str, _TokenSettings] = {}
    for entry in input_tokens:
        token_settings[entry.get("name", "")] = entry["settings"]

    filled_tokens: list[dict[str, object]] = []
    todo_tokens: list[dict[str, object]] = []
    for name_, scopes_dict in token_keys.items():
        real_scopes: list[str] = [s for s in scopes_dict if s != "global_scope"]
        all_props: set[str] = {p for props in scopes_dict.values() for p in props}
        src: _TokenSettings | None = token_settings.get(name_)
        # Skip colorless entries not found in the input — they carry no useful value.
        if src is None and not all_props:
            continue
        settings: dict[str, str] = {}
        for prop_ in all_props:
            val: str | None = None
            if src is not None:
                if prop_ == "foreground":
                    val = src.get("foreground")
                elif prop_ == "background":
                    val = src.get("background")
                elif prop_ == "fontStyle":
                    val = src.get("fontStyle")
            settings[prop_] = val if val is not None else placeholder
        token_entry: dict[str, object] = {"settings": settings}
        if name_:
            token_entry["name"] = name_
        if real_scopes:
            token_entry["scope"] = real_scopes[0] if len(real_scopes) == 1 else real_scopes
        # Entries with no source in the input go into a /* */ comment block.
        if src is None:
            todo_tokens.append(token_entry)
        else:
            filled_tokens.append(token_entry)

    out: dict[str, object] = {k: v for k, v in data.items() if k != "tokenColors"}
    out["colors"] = out_colors
    out["semanticTokenColors"] = filled_semantic
    out["tokenColors"] = filled_tokens

    placeholder_value: str = json.dumps(placeholder)
    lines: list[str] = json.dumps(out, indent=2).splitlines()
    suffix: str = f": {placeholder_value}"
    annotated: list[str] = [
        f"// {line}" if line.rstrip().endswith(f"{suffix},") or line.rstrip().endswith(suffix) else line
        for line in lines
    ]
    result_lines: list[str] = list(annotated)

    # Insert /* */ comment blocks for todo semantic entries inside semanticTokenColors.
    if todo_semantic:
        sem_block: list[str] = []
        for sem_tok, sem_val in todo_semantic:
            entry_str = json.dumps({sem_tok: sem_val}, indent=2)  # type: ignore[misc]
            # Strip the outer { } to get just the "key": value lines.
            inner_lines = entry_str.splitlines()[1:-1]
            if len(inner_lines) == 1:
                sem_block.append("  //" + inner_lines[0])
            else:
                sem_block.append("  /*")
                for el in inner_lines:
                    sem_block.append("  " + el)
                sem_block.append("  */")

        final_lines2: list[str] = []
        in_sc = False
        for line in result_lines:
            if not in_sc and '"semanticTokenColors"' in line:
                in_sc = True
                if "{}" in line:
                    sc_content = "\n".join(sem_block)
                    final_lines2.append(line.replace("{}", "{\n" + sc_content + "\n  }"))
                    in_sc = False
                    continue
            if in_sc and line.rstrip(",") == "  }":
                final_lines2.extend(sem_block)
                in_sc = False
            final_lines2.append(line)
        result_lines = final_lines2

    # Insert /* */ comment blocks for todo token entries inside the tokenColors array.
    if todo_tokens:
        todo_block: list[str] = []
        for todo_entry in todo_tokens:
            todo_block.append("    /*")
            for el in json.dumps(todo_entry, indent=2).splitlines():
                todo_block.append("    " + el)
            todo_block.append("    */")

        final_lines: list[str] = []
        in_tc = False
        for line in result_lines:
            if not in_tc and '"tokenColors"' in line:
                in_tc = True
                if "[]" in line:
                    tc_content = "\n".join(todo_block)
                    final_lines.append(line.replace("[]", "[\n" + tc_content + "\n  ]"))
                    in_tc = False
                    continue
            if in_tc and line.rstrip(",") == "  ]":
                final_lines.extend(todo_block)
                in_tc = False
            final_lines.append(line)
        result_lines = final_lines

    # Final pass: strip trailing commas left on lines that precede only comment
    # content (// lines or /* */ blocks) before the next structural closer.
    def _is_comment(ln: str) -> bool:
        s = ln.strip()
        return s.startswith("//") or s in ("/*", "*/")

    fixed: list[str] = []
    in_blk = False
    for i, line in enumerate(result_lines):
        s = line.strip()
        if s == "/*":
            in_blk = True
        elif s == "*/":
            in_blk = False
        if not in_blk and not _is_comment(line) and line.rstrip().endswith(","):
            j, jb = i + 1, False
            while j < len(result_lines):
                sj = result_lines[j].strip()
                if sj == "/*":
                    jb = True
                elif sj == "*/":
                    jb = False
                elif not jb and not sj.startswith("//"):
                    break
                j += 1
            if j < len(result_lines) and result_lines[j].strip() in ("}", "},", "]", "],"):
                fixed.append(line.rstrip()[:-1])
                continue
        fixed.append(line)
    result_lines = fixed

    has_placeholders: bool = bool(todo or todo_tokens or todo_semantic)
    stem: str = colorscheme_file.stem
    ext: str = ".json5" if has_placeholders else ".json"
    output_dir.mkdir(exist_ok=True)
    out_path: Path = output_dir / (stem + ext)
    out_path.write_text("\n".join(result_lines) + "\n")
    typer.echo(f"{out_path}: {filled} filled, {todo} {placeholder!r} placeholders")


if __name__ == "__main__":
    app()

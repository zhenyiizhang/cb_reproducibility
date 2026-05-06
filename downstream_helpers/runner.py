from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def list_output_files(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file())


def display_svg_outputs(svg_paths: Sequence[Path]) -> None:
    from IPython.display import SVG, display

    for path in svg_paths:
        if path.exists():
            print(path.name)
            display(SVG(filename=str(path)))


def display_image_outputs(image_paths: Sequence[Path]) -> None:
    from IPython.display import Image, display

    for path in image_paths:
        if path.exists():
            print(path.name)
            display(Image(filename=str(path)))


def display_html_outputs(html_paths: Iterable[Path], *, height: int = 900) -> None:
    from IPython.display import HTML, IFrame, display

    for path in html_paths:
        if path.exists():
            print(path.name)
            resolved = path.resolve()
            display(IFrame(src=resolved.as_uri(), width="100%", height=height))
            display(HTML(f'<p><a href="{resolved.as_uri()}" target="_blank">Open {path.name} in a new tab</a></p>'))


def export_plotly_figure(
    fig,
    *,
    svg_path: Path | None = None,
    pdf_path: Path | None = None,
    png_path: Path | None = None,
    vector_scale: float = 1.0,
    png_scale: float = 2.0,
) -> None:
    import plotly.io as pio

    if svg_path is not None:
        pio.write_image(fig, str(svg_path), scale=vector_scale)
    if pdf_path is not None:
        pio.write_image(fig, str(pdf_path), scale=vector_scale)
    if png_path is not None:
        pio.write_image(fig, str(png_path), scale=png_scale)

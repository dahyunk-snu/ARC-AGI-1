from pathlib import Path

import numpy as np

from rich.console import Console
from rich.text import Text
from typing import List

color_map = {
    0: "black",
    1: "red",
    2: "green",
    3: "yellow",
    4: "blue",
    5: "magenta",
    6: "cyan",
    7: "white",
    8: "bright_red",
    9: "bright_green",
}

console = Console()

def make_rich_lines(grid: List[List[int]]) -> List[Text]:
    lines = []
    for row in grid:
        visual = Text()
        for cell in row:
            color = color_map.get(cell, "white")
            visual.append("  ", style=f"on {color}")
        raw = Text("  " + str(row))
        visual.append(raw)
        lines.append(visual)
    return lines

def render_grid(grid: List[List[int]]):
    lines = make_rich_lines(grid)
    for line in lines:
        console.print(line)

def make_rich_lines_for_parallel(grid: List[List[int]], cell_width: int = 4) -> List[Text]:
    lines = []
    for row in grid:
        visual = Text()
        for cell in row:
            color = color_map.get(cell, "white")
            visual.append(" " * cell_width, style=f"on {color}")
        lines.append(visual)
    return lines

def render_grids_parallel(grids: List[List[List[int]]], cell_width: int = 3, spacing: int = 4):
    rich_grids = [make_rich_lines_for_parallel(grid, cell_width) for grid in grids]

    max_rows = max(len(lines) for lines in rich_grids)

    for lines in rich_grids:
        while len(lines) < max_rows:
            lines.append(Text(" " * (len(lines[0]) if lines else 0))) 

    for row_index in range(max_rows):
        combined = Text()
        for lines in rich_grids:
            combined += lines[row_index]
            combined += Text(" " * spacing)
        console.print(combined)

def get_base_model(model_name):
    available_models = [
        "meta-llama/Llama-3.2-1B",
    ]
    assert model_name in available_models, f"{model_name} is not available."


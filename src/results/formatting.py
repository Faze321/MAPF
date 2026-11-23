from __future__ import annotations

import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    for col in display.columns:
        formatted_values = []
        for value in display[col]:
            if pd.isna(value):
                formatted_values.append("")
                continue
            text = str(value).replace("\n", " ")
            formatted_values.append(text[:220] + "..." if len(text) > 223 else text)
        display[col] = formatted_values
    widths = {
        col: max(len(str(col)), *(len(str(value)) for value in display[col]))
        for col in display.columns
    }
    header = "| " + " | ".join(str(col).ljust(widths[col]) for col in display.columns) + " |"
    separator = "| " + " | ".join("-" * widths[col] for col in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(row[col]).ljust(widths[col]) for col in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *rows]) + "\n"

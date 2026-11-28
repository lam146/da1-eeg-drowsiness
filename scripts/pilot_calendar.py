"""Resolve the N16 measurement date before grouping recordings for validation."""
from pathlib import Path
import re
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd


def measurement_date(note, filename):
    with ZipFile(note) as doc:
        root = ElementTree.fromstring(doc.read("word/document.xml"))
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = ["".join(t.text or "" for t in p.iter(ns + "t")) for p in root.iter(ns + "p")]
    dates = set(re.findall(r"Ngày đo\s*:\s*\[?\s*(\d{1,2}/\d{1,2}/\d{4})", "\n".join(paragraphs), re.I))
    if len(dates) > 1:
        raise ValueError(f"Conflicting measurement dates in {note}")
    if dates:
        return pd.to_datetime(dates.pop(), format="%d/%m/%Y"), "note"
    match = re.search(r"_(\d{8})_", Path(filename).name)
    if not match:
        raise ValueError(f"Missing measurement date: {filename}")
    return pd.to_datetime(match[1], format="%d%m%Y"), "filename"

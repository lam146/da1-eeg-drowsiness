from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def test_measurement_day_uses_note_even_when_word_splits_the_date(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from pilot_calendar import measurement_date

    doc = tmp_path / "02.docx"
    with ZipFile(doc, "w") as z:
        z.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   '<w:p><w:t>Ngày đo: [26/</w:t><w:t>10/2026]</w:t></w:p></w:document>')
    date, source = measurement_date(doc, "S01_25102026_PILOT_02.csv")
    assert str(date.date()) == "2026-10-26" and source == "note"


def test_personal_mixture_combines_margins_across_different_scalers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate_repeat_wearers import FEATURES, variants, fit

    rng = np.random.default_rng(16)
    donor = pd.DataFrame(rng.normal(size=(60,len(FEATURES))),columns=FEATURES)
    donor = donor.assign(subject_id="S01",session_id="A",label=(donor[FEATURES[0]] > 0).astype(int))
    own = donor.assign(subject_id="S02", session_id="B").copy()
    own[FEATURES] = 2 * own[FEATURES] + 3
    models = variants(donor,fit(donor),own)
    def margin(model):
        return ((own[FEATURES].to_numpy()-model.mean)/model.scale) @ model.weights + model.bias
    np.testing.assert_allclose(margin(models["pooled_0.25"]),
                               .75*margin(models["cold"])+.25*margin(models["pooled"]))

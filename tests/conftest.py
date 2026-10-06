import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
UVPD = ROOT / "data" / "raw" / "lipidoracle" / "extracted" / "04_uvpd"
WEIGHTS = ROOT / "weights"

# The single spectrum used for every forward/backward test (IMPLEMENTATION.md, "Test spectrum").
TEST_TITLE_NAME = "PC 15:0/18:1"
TEST_ADDUCT = "[M+H]+"


@pytest.fixture(scope="session")
def uvpd_records():
    if not UVPD.exists():
        pytest.skip("LipidOracle UVPD data not downloaded (see IMPLEMENTATION.md)")
    from duo_lipa.data.lipidoracle import load_uvpd

    return load_uvpd(UVPD)


@pytest.fixture(scope="session")
def test_record(uvpd_records):
    """First EquiSPLASH PC 15:0_18:1;d7 [M+H]+ spectrum; label joins to PC 15:0/18:1(9Z)."""
    for r in uvpd_records:
        if r.meta["title_name"].startswith("PC 15:0_18:1") and r.adduct == TEST_ADDUCT:
            return r
    raise RuntimeError("test spectrum not found")


@pytest.fixture(scope="session")
def schema():
    from duo_lipa.schema.schema import SlotSchema

    return SlotSchema()

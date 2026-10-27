"""Shared settings: which indicators to use, where files live, and which codes to drop."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# World Bank World Development Indicators (WDI).
# Downloaded from the open-data mirror at github.com/datasets/world-development-indicators,
# which re-publishes the official World Bank series as CSV files and refreshes them monthly.
MIRROR_URL = (
    "https://raw.githubusercontent.com/datasets/world-development-indicators/"
    "main/indicators/{code}/data.csv"
)

# short name -> WDI indicator code
INDICATORS = {
    "inflation": "fp.cpi.totl.zg",        # Consumer price inflation, annual %  (the target)
    "deflator": "ny.gdp.defl.kd.zg",      # GDP deflator inflation, annual %
    "gdp_growth": "ny.gdp.mktp.kd.zg",    # Real GDP growth, annual %
    "unemployment": "sl.uem.totl.zs",     # Unemployment rate, % of labor force (ILO modeled)
    "money_growth": "fm.lbl.bmny.zg",     # Broad money (M2) growth, annual %
    "fx_rate": "pa.nus.fcrf",             # Official exchange rate, local currency per US$
    "lending_rate": "fr.inr.lend",        # Bank lending interest rate, %
    "real_rate": "fr.inr.rinr",           # Real interest rate, %
    "imports_gdp": "ne.imp.gnfs.zs",      # Imports of goods & services, % of GDP
}

# The WDI files mix real countries with regional / income-group aggregates
# ("World", "Euro area", "Low income", ...). Aggregates are averages of countries
# that are already in the data, so keeping them would double-count.
AGGREGATE_CODES = {
    "AFE", "AFW", "ARB", "CEB", "CSS", "EAP", "EAR", "EAS", "ECA", "ECS", "EMU",
    "EUU", "FCS", "HIC", "HPC", "IBD", "IBT", "IDA", "IDB", "IDX", "INX", "LAC",
    "LCN", "LDC", "LIC", "LMC", "LMY", "LTE", "MEA", "MIC", "MNA", "NAC", "OED",
    "OSS", "PRE", "PSS", "PST", "SAS", "SSA", "SSF", "SST", "TEA", "TEC", "TLA",
    "TMN", "TSA", "TSS", "UMC", "WLD",
}

START_YEAR = 1990          # ILO unemployment estimates begin in 1991
FIRST_TEST_YEAR = 2006     # backtest forecasts target years 2006 .. latest
RANDOM_STATE = 42

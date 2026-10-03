"""Nifty 500 universe handling.

Two sources:
  * get_nifty500_live()   -- fetch the CURRENT official constituents from NSE.
                             Works where nseindia.com is reachable (your
                             machine); used by the yfinance path.
  * NIFTY500_BUNDLED      -- a dated, best-effort fallback list of real NSE
                             symbols, used when the live fetch is unreachable
                             (e.g. this sandbox). VERIFY/refresh against NSE.

SURVIVORSHIP BIAS WARNING (read this):
  Both sources give only *current* constituents. This strategy deliberately
  buys stocks cratering to fresh 52-week lows -- precisely the population where
  some names keep falling, get removed from the index, or are delisted. A
  current-constituents universe has already dropped those losers, so the win
  rate and returns measured on it are biased UPWARD. Treat any positive result
  as an optimistic upper bound, not a realistic expectation. A truly honest
  backtest needs a point-in-time (survivorship-bias-free) constituent history,
  which neither yfinance nor a current index list provides.
"""
from __future__ import annotations

BUNDLED_AS_OF = "2025 (approx, verify against NSE)"

# Live fetch of the official list (for environments with internet).
NSE_NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"


def get_nifty500_live(timeout: int = 20) -> list[str]:
    """Fetch current Nifty 500 trading symbols from NSE. Raises on failure."""
    import io
    import requests
    import pandas as pd
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    r = requests.get(NSE_NIFTY500_URL, headers=headers, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # column is "Symbol"
    col = "Symbol" if "Symbol" in df.columns else df.columns[2]
    return [str(s).strip().upper() for s in df[col].dropna().tolist()]


# A best-effort, dated snapshot of real NSE symbols that are / were Nifty 500
# members. Not guaranteed exhaustive or exactly current; symbols that do not
# resolve on the data source are skipped and reported. Refresh with the live
# fetch for an authoritative list.
NIFTY500_BUNDLED = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT",
    "MARUTI", "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "NESTLEIND",
    "ONGC", "NTPC", "POWERGRID", "M&M", "TATAMOTORS", "TATASTEEL", "JSWSTEEL",
    "ADANIENT", "ADANIPORTS", "COALINDIA", "BAJAJFINSV", "HDFCLIFE", "SBILIFE",
    "GRASIM", "BRITANNIA", "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT", "HEROMOTOCO",
    "BAJAJ-AUTO", "INDUSINDBK", "TECHM", "APOLLOHOSP", "HINDALCO", "TATACONSUM",
    "UPL", "BPCL", "IOC", "GAIL", "VEDL", "DABUR", "GODREJCP", "MARICO", "PIDILITIND",
    "BERGEPAINT", "COLPAL", "HAVELLS", "DLF", "GODREJPROP", "OBEROIRLTY", "AMBUJACEM",
    "ACC", "SHREECEM", "DMART", "ADANIGREEN", "ADANIPOWER", "ADANIENSOL", "SIEMENS",
    "ABB", "BOSCHLTD", "BEL", "HAL", "BHEL", "SAIL", "NMDC", "NATIONALUM", "JINDALSTEL",
    "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "IDFCFIRSTB", "FEDERALBNK", "BANDHANBNK",
    "AUBANK", "INDHOTEL", "TRENT", "PAGEIND", "PGHH", "GILLETTE", "3MINDIA", "HONAUT",
    "ABBOTINDIA", "ALKEM", "AUROPHARMA", "LUPIN", "BIOCON", "TORNTPHARM", "ZYDUSLIFE",
    "GLENMARK", "IPCALAB", "LAURUSLABS", "MANKIND", "GLAND", "SYNGENE", "NAUKRI",
    "ZOMATO", "PAYTM", "POLICYBZR", "NYKAA", "IRCTC", "INDIGO", "INTERGLOBE",
    "MOTHERSON", "BALKRISIND", "MRF", "APOLLOTYRE", "CEAT", "BHARATFORG", "TIINDIA",
    "ESCORTS", "ASHOKLEY", "TVSMOTOR", "EXIDEIND", "AMARAJABAT", "SUNDRMFAST",
    "SCHAEFFLER", "SKFINDIA", "CUMMINSIND", "THERMAX", "AIAENG", "POLYCAB", "KEI",
    "VOLTAS", "BLUESTARCO", "WHIRLPOOL", "CROMPTON", "DIXON", "AMBER", "VGUARD",
    "BATAINDIA", "RELAXO", "VBL", "UBL", "RADICO", "PGHL", "EMAMILTD", "JUBLFOOD",
    "DEVYANI", "WESTLIFE", "TATACHEM", "DEEPAKNTR", "AARTIIND", "SRF", "NAVINFLUOR",
    "ATUL", "VINATIORGA", "FLUOROCHEM", "PIIND", "COROMANDEL", "CHAMBLFERT", "GNFC",
    "GSPL", "IGL", "MGL", "PETRONET", "GUJGASLTD", "OIL", "HINDPETRO", "CASTROLIND",
    "MPHASIS", "LTIM", "PERSISTENT", "COFORGE", "LTTS", "TATAELXSI", "KPITTECH",
    "CYIENT", "BSOFT", "OFSS", "ZENSARTECH", "FSL", "SONACOMS", "INDUSTOWER",
    "TATACOMM", "ROUTE", "TANLA", "MUTHOOTFIN", "MANAPPURAM", "CHOLAFIN", "SHRIRAMFIN",
    "SUNDARMFIN", "LICHSGFIN", "PNBHOUSING", "CANFINHOME", "AAVAS", "HOMEFIRST",
    "ICICIGI", "ICICIPRULI", "STARHEALTH", "GICRE", "NIACL", "MAXHEALTH", "FORTIS",
    "LALPATHLAB", "METROPOLIS", "KIMS", "NH", "RAINBOW", "POONAWALLA", "ABCAPITAL",
    "BAJAJHLDNG", "CHOLAHLDNG", "PEL", "RECLTD", "PFC", "IRFC", "IREDA", "HUDCO",
    "SJVN", "NHPC", "TORNTPOWER", "CESC", "JSWENERGY", "TATAPOWER", "NLCINDIA",
    "APLAPOLLO", "RATNAMANI", "WELCORP", "JINDALSAW", "HINDZINC", "GRAVITA", "APARINDS",
    "FINCABLES", "CGPOWER", "HITACHIEN", "GET&D", "INOXWIND", "SUZLON", "KALPATPOWR",
    "KEC", "RVNL", "IRCON", "NBCC", "RITES", "ENGINERSIN", "NCC", "PNCINFRA",
    "KNRCON", "GRINFRA", "ASHOKA", "JKCEMENT", "RAMCOCEM", "DALBHARAT", "NUVOCO",
    "JKLAKSHMI", "HEIDELBERG", "STARCEMENT", "BIRLACORPN", "INDIACEM", "ASTRAL",
    "SUPREMEIND", "FINPIPE", "PRINCEPIPE", "KAJARIACER", "CERA", "HSCL", "CENTURYPLY",
    "GREENPANEL", "GREENLAM", "SAREGAMA", "PVRINOX", "SUNTV", "ZEEL", "NETWORK18",
    "TV18BRDCST", "DBCORP", "JAGRAN", "NAVNETEDUL", "HATHWAY", "GSFC", "RCF",
    "FACT", "DEEPAKFERT", "MANGCHEFER", "NFL", "BAYERCROP", "RALLIS", "SUMICHEM",
    "DHANUKA", "INSECTICID", "SHARDACROP", "UPL", "BASF", "LINDEINDIA", "GHCL",
    "TATAINVEST", "JIOFIN", "IIFL", "IIFLWAM", "360ONE", "ANGELONE", "CDSL",
    "BSE", "MCX", "CAMS", "KFINTECH", "NAM-INDIA", "HDFCAMC", "UTIAMC", "ABSLAMC",
    "CREDITACC", "FIVESTAR", "SPANDANA", "UJJIVANSFB", "EQUITASBNK", "RBLBANK",
    "CUB", "KARURVYSYA", "SOUTHBANK", "J&KBANK", "DCBBANK", "CSBBANK", "TMB",
    "YESBANK", "IDBI", "UCOBANK", "CENTRALBK", "IOB", "MAHABANK", "PSB", "INDIANB",
    "ABFRL", "ADANITRANS", "AEGISCHEM", "AFFLE", "AJANTPHARM", "APLLTD", "ARE&M",
    "ASTERDM", "ASTRAZEN", "BALAMINES", "BALRAMCHIN", "BANKINDIA", "BBTC", "BEML",
    "BIKAJI", "BLUEDART", "BORORENEW", "BRIGADE", "CAMPUS", "CAPLIPOINT", "CARBORUNIV",
    "CCL", "CHALET", "CHENNPETRO", "CLEAN", "CMSINFO", "CONCOR", "CONCORDBIO",
    "CRAFTSMAN", "CRISIL", "DATAPATTNS", "DCAL", "DELHIVERY", "DOMS", "EIDPARRY",
    "ELECON", "ELGIEQUIP", "EMCURE", "ENDURANCE", "EPL", "ERIS", "FINEORG",
    "GESHIP", "GMMPFAUDLR", "GMRINFRA", "GOCOLORS", "GRANULES", "GRINDWELL",
    "GPIL", "GPPL", "HBLPOWER", "HFCL", "HAPPSTMNDS", "IEX", "IFBIND", "INDIAMART",
    "INDIGOPNTS", "INTELLECT", "IRB", "ISEC", "JBCHEPHARM", "JBMA", "JKPAPER",
    "JWL", "JYOTHYLAB", "KANSAINER", "KAYNES", "KPRMILL", "KSB", "LEMONTREE",
    "LTFOODS", "MAHSEAMLES", "MAPMYINDIA", "MASTEK", "MEDANTA", "MEDPLUS",
    "METROBRAND", "MINDACORP", "MOTILALOFS", "NATCOPHARM", "NEWGEN", "NUVAMA",
    "OLECTRA", "PGEL", "PNCINFRA", "PPLPHARMA", "PRESTIGE", "RAILTEL", "RAJESHEXPO",
    "RATEGAIN", "RAYMOND", "REDINGTON", "RHIM", "RKFORGE", "RTNINDIA", "SAPPHIRE",
    "SFL", "SHARDAMOTR", "SHYAMMETL", "SIGNATURE", "SOBHA", "SONATSOFTW", "SUMITOMO",
    "SUNTECK", "SUPRAJIT", "SWSOLAR", "SYMPHONY", "TATATECH", "TEJASNET", "TITAGARH",
    "TRIDENT", "TRITURBINE", "TTKPRESTIG", "UNOMINDA", "USHAMART", "VAIBHAVGBL",
    "VARROC", "VIJAYA", "VTL", "WELSPUNLIV", "ZENSARTECH", "ZFCVINDIA",
]
# de-duplicate while preserving order
NIFTY500_BUNDLED = list(dict.fromkeys(NIFTY500_BUNDLED))


def get_universe(prefer_live: bool = True) -> tuple[list[str], str]:
    """Return (symbols, source_label). Falls back to the bundled list."""
    if prefer_live:
        try:
            syms = get_nifty500_live()
            if len(syms) >= 100:
                return syms, "NSE live ind_nifty500list.csv"
        except Exception as e:
            print(f"  [universe] live fetch failed ({e!r}); using bundled list")
    return NIFTY500_BUNDLED, f"bundled snapshot ({BUNDLED_AS_OF})"

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# ========= CONFIG =========
SYMBOLS = {
    "SPCX": "Space Exploration Technologies Corp",
    "MU": "Micron Technology",
    "NVDA": "NVIDIA",
    "SNDK": "SanDisk",
    "TSLA": "Tesla",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel",
    "MRVL": "Marvell Technology",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AVGO": "Broadcom",
    "META": "Meta Platforms",
    "AMZN": "Amazon",
    "AMAT": "Applied Materials",
    "GOOG": "Alphabet Class C",
    "WDC": "Western Digital",
    "STX": "Seagate Technology",
    "PLTR": "Palantir",
    "LRCX": "Lam Research",
    "LITE": "Lumentum Holdings",
    "KLAC": "KLA Corporation",
    "NBIS": "Nebius Group",
    "APP": "AppLovin",
    "ORCL": "Oracle",
    "ROKU": "Roku",
    "HOOD": "Robinhood Markets",
    "LLY": "Eli Lilly",
    "GEV": "GE Vernova",
    "XOM": "Exxon Mobil",
    "RKLB": "Rocket Lab",
    "QCOM": "Qualcomm",
    "NFLX": "Netflix",
    "DELL": "Dell Technologies",
    "AAOI": "Applied Optoelectronics",
    "AAL": "American Airlines",
    "UNH": "UnitedHealth Group",
    "CSCO": "Cisco Systems",
    "MSTR": "Strategy Inc.",
    "JPM": "JPMorgan Chase",
    "GS": "Goldman Sachs",
    "BE": "Bloom Energy",
    "WMT": "Walmart",
    "NOW": "ServiceNow",
    "CRWV": "CoreWeave",
    "TXN": "Texas Instruments",
    "CAT": "Caterpillar",
    "CRM": "Salesforce",
    "V": "Visa",
    "COHR": "Coherent",
    "SATS": "EchoStar",
    "PANW": "Palo Alto Networks",
    "ASTS": "AST SpaceMobile",
    "ALAB": "Astera Labs",
    "CRDO": "Credo Technology Group",
    "IREN": "IREN Limited",
    "ADBE": "Adobe",
    "TER": "Teradyne",
    "BA": "Boeing",
    "ADI": "Analog Devices",
    "IBM": "IBM",
    "MA": "Mastercard",
    "SMCI": "Super Micro Computer",
    "HON": "Honeywell",
    "COST": "Costco",
    "CVX": "Chevron",
    "CIEN": "Ciena",
    "JNJ": "Johnson & Johnson",
    "C": "Citigroup",
    "UBER": "Uber",
    "GE": "GE Aerospace",
    "BAC": "Bank of America",
    "MRK": "Merck",
    "CRWD": "CrowdStrike",
    "RGNT": "Regentis Biomaterials",
    "GLW": "Corning",
    "VRT": "Vertiv",
    "BKNG": "Booking Holdings",
    "KO": "Coca-Cola",
    "IONQ": "IonQ",
    "HD": "Home Depot",
    "AMGN": "Amgen",
    "COIN": "Coinbase",
    "HPE": "Hewlett Packard Enterprise",
    "LIN": "Linde",
    "INTU": "Intuit",
    "APH": "Amphenol",
    "SNOW": "Snowflake",
    "COF": "Capital One",
    "ON": "ON Semiconductor",
    "PG": "Procter & Gamble",
    "DDOG": "Datadog",
    "AXTI": "AXT Inc.",
    "PEP": "PepsiCo",
    "SOFI": "SoFi Technologies",
    "CRCL": "Circle Internet Group",
    "FISV": "Fiserv",
    "MS": "Morgan Stanley",
    "MPWR": "Monolithic Power Systems",
    "MCD": "McDonald's",
    "ANET": "Arista Networks",
}

LOOKBACK = "400d"
INTERVAL = "1d"
TRADING_DAYS = 252

OUT_JSON = "data/marketdata.json"
OUT_CSV = None  # ex.: "data/marketdata.csv"


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _append_nulls(results, batch):
    """Se um batch falhar, adiciona linhas com null para não quebrar o pipeline."""
    for sym in batch:
        # Busca o nome no dicionário SYMBOLS
        company_name = SYMBOLS.get(sym, sym) 
        
        results.append(
            {
                "symbol": sym,
                "name": company_name, 
                "price": None,
                "vol_annual": None,
            }
        )


def _extract_close_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai a matriz de fechamentos (Close) no formato:
      index = datas
      colunas = tickers
    Suporta o formato que o yfinance devolve para 1 ticker ou vários tickers.
    """
    # Caso o df venha vazio
    if df is None or df.empty:
        return pd.DataFrame()

    # Caso comum com group_by="column": df["Close"] funciona se existir
    if "Close" in df.columns:
        close = df["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame()
        return close

    # Caso alternativo: MultiIndex nas colunas (ex.: ('Close', 'PETR4.SA'))
    if isinstance(df.columns, pd.MultiIndex):
        # tenta achar o nível "Close"
        if "Close" in df.columns.get_level_values(0):
            close = df.xs("Close", axis=1, level=0, drop_level=True)
            if isinstance(close, pd.Series):
                close = close.to_frame()
            return close

    # Se não conseguiu extrair
    return pd.DataFrame()


def main():
    os.makedirs("data", exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    results = []

    for batch in chunked(list(SYMBOLS.keys()), 100):
        try:
            df = yf.download(
                tickers=batch,
                period=LOOKBACK,
                interval=INTERVAL,
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="column",
            )
        except Exception as e:
            print(f"[WARN] Batch falhou (exception): {e}")
            _append_nulls(results, batch)
            continue

        close = _extract_close_df(df)

        # Se não veio nada de Close / veio vazio, não quebra: registra null e segue
        if close.empty or len(close.index) == 0:
            print("[WARN] Batch retornou vazio/sem Close. Registrando nulls.")
            _append_nulls(results, batch)
            continue

        # Garante que todas as colunas estejam presentes (se alguns tickers falharam)
        # cria colunas faltantes com NaN
        for sym in batch:
            if sym not in close.columns:
                close[sym] = np.nan

        # Ordena colunas para consistência (opcional)
        close = close[batch]

        # Último preço conhecido por ticker
        close_ffill = close.ffill()
        # Se por algum motivo ainda não tiver linha depois do ffill, evita iloc[-1]
        if close_ffill.empty or len(close_ffill.index) == 0:
            print("[WARN] Close após ffill ficou vazio. Registrando nulls.")
            _append_nulls(results, batch)
            continue

        last_price = close_ffill.iloc[-1]

        # Vol anualizada (log-retornos)
        logret = np.log(close / close.shift(1))
        vol = logret.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)

        for sym in batch:
            p = last_price.get(sym, np.nan)
            v = vol.get(sym, np.nan)
            
            # Pega do cache, ou usa o ticker se não achar
            company_name = SYMBOLS.get(sym, sym)

            results.append(
                {
                    "symbol": sym,
                    "name": company_name, 
                    "price": None if pd.isna(p) else float(round(float(p), 6)),
                    "vol_annual": None if pd.isna(v) else float(round(float(v), 8)),
                }
            )

    payload = {
        "generated_at_utc": now,
        "source": "yfinance",
        "interval": INTERVAL,
        "lookback": LOOKBACK,
        "trading_days": TRADING_DAYS,
        "count": len(results),
        "data": results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if OUT_CSV:
        pd.DataFrame(results).to_csv(OUT_CSV, index=False, encoding="utf-8")

    ok_prices = sum(1 for r in results if r["price"] is not None)
    ok_vols = sum(1 for r in results if r["vol_annual"] is not None)
    print(f"OK: atualizado {OUT_JSON} com {len(results)} tickers.")
    print(f"   Preços OK: {ok_prices} | Vols OK: {ok_vols}")


if __name__ == "__main__":
    main()

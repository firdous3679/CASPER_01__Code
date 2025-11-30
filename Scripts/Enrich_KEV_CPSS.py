#!/usr/bin/env python3
import io, sys, time, math, argparse, os, requests, pandas as pd
from pathlib import Path

INPUT_CSV  = Path("power_grid_cves.csv")
OUTPUT_CSV = Path("power_grid_cves_enriched.csv")

# Default network endpoints (override with --kev_csv / --epss_csv to use local files)
CISA_KEV_CSV_URL = "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
EPSS_API_URL     = "https://api.first.org/data/v1/epss"  # ?cve=CVE-...,CVE-...

# ---------- utils ----------
def normalize_cve_id(x: str) -> str:
    if not isinstance(x, str): return ""
    x = x.strip().upper()
    # basic cleanup
    if x.startswith("CVE-"):
        return x
    # sometimes IDs come as “cve: CVE-2021-1234”
    parts = [p for p in x.split() if p.startswith("CVE-")]
    return parts[0] if parts else x

def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr); sys.exit(2)
    df = pd.read_csv(path, dtype=str)
    # normalize column name
    if "cve_id" not in df.columns:
        for c in df.columns:
            if c.strip().lower() in ("cve", "cveid", "cve_id"):
                df = df.rename(columns={c: "cve_id"})
                break
    if "cve_id" not in df.columns:
        print("ERROR: Input CSV must have a 'cve_id' column.", file=sys.stderr); sys.exit(2)
    df["cve_id"] = df["cve_id"].map(normalize_cve_id)
    return df

# ---------- KEV ----------
def load_kev(kev_csv: str | None) -> pd.DataFrame:
    """
    Load KEV CSV either from URL or local path.
    Expected columns include at least: cveID, dateAdded
    """
    if kev_csv:
        p = Path(kev_csv)
        if not p.exists():
            print(f"ERROR: KEV CSV not found at {kev_csv}", file=sys.stderr); sys.exit(2)
        kev = pd.read_csv(p, dtype=str)
    else:
        r = requests.get(CISA_KEV_CSV_URL, timeout=60)
        r.raise_for_status()
        kev = pd.read_csv(io.BytesIO(r.content), dtype=str)

    # normalize
    cols = {c.lower(): c for c in kev.columns}
    if "cveid" not in cols:
        raise RuntimeError("KEV CSV missing cveID column.")
    cve_col = cols["cveid"]
    date_col = cols.get("dateadded")

    out = pd.DataFrame({
        "cve_id": kev[cve_col].astype(str).map(normalize_cve_id)
    })
    if date_col:
        out["kev_added_dt"] = kev[date_col]
    else:
        out["kev_added_dt"] = pd.NA
    out["in_cisa_kev"] = True
    out = out.drop_duplicates(subset=["cve_id"])
    return out

# ---------- EPSS ----------
def fetch_epss_batch(cves: list[str], retries=3, sleep=0.6) -> pd.DataFrame:
    """
    Query EPSS for a list of up to ~200 CVEs. If 414/431, we'll return empty and the caller will split smaller.
    """
    if not cves:
        return pd.DataFrame(columns=["cve_id","epss_score","epss_percentile","epss_date"])
    q = ",".join(cves)
    for attempt in range(retries):
        try:
            resp = requests.get(EPSS_API_URL, params={"cve": q}, timeout=60)
            if resp.status_code in (414, 431):
                # Let caller split the list smaller
                return pd.DataFrame()
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                return pd.DataFrame(columns=["cve_id","epss_score","epss_percentile","epss_date"])
            df = pd.DataFrame(data)
            # normalize
            rename = {}
            if "cve" in df.columns: rename["cve"] = "cve_id"
            if "epss" in df.columns: rename["epss"] = "epss_score"
            if "percentile" in df.columns: rename["percentile"] = "epss_percentile"
            if "date" in df.columns: rename["date"] = "epss_date"
            if "created" in df.columns and "epss_date" not in rename: rename["created"] = "epss_date"
            df = df.rename(columns=rename)
            for col in ("epss_score","epss_percentile"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df["cve_id"] = df["cve_id"].astype(str).map(normalize_cve_id)
            return df[["cve_id","epss_score","epss_percentile","epss_date"]].drop_duplicates("cve_id")
        except Exception as e:
            if attempt + 1 == retries:
                print(f"EPSS request failed for {len(cves)} CVEs: {e}", file=sys.stderr)
                return pd.DataFrame()
            time.sleep(sleep)

def fetch_epss_recursive(cves: list[str], max_batch=200) -> pd.DataFrame:
    """
    Robust EPSS fetching:
      - splits into batches of <= max_batch
      - on 414/empty result, splits that batch into halves recursively so no CVEs are lost
    """
    cves = [normalize_cve_id(x) for x in pd.unique(pd.Series(cves).dropna()) if x.startswith("CVE-")]
    if not cves:
        return pd.DataFrame(columns=["cve_id","epss_score","epss_percentile","epss_date"])

    results = []
    def process(group: list[str]):
        if not group:
            return
        if len(group) > max_batch:
            mid = len(group)//2
            process(group[:mid]); process(group[mid:]); return
        df = fetch_epss_batch(group)
        if df is None or df.empty:
            if len(group) == 1:
                # give up on this one CVE
                return
            # split smaller to avoid 414 or payload quirks
            mid = len(group)//2
            process(group[:mid]); process(group[mid:])
            return
        results.append(df)

    process(cves)
    if not results:
        return pd.DataFrame(columns=["cve_id","epss_score","epss_percentile","epss_date"])
    return pd.concat(results, ignore_index=True).drop_duplicates(subset=["cve_id"])

def load_epss_from_csv(path: str) -> pd.DataFrame:
    """
    Offline EPSS CSV loader. Expected columns at minimum: cve (or cve_id), epss, percentile.
    Many public EPSS CSVs include headers like: cve, epss, percentile, date
    """
    p = Path(path)
    if not p.exists():
        print(f"ERROR: EPSS CSV not found at {path}", file=sys.stderr); sys.exit(2)
    df = pd.read_csv(p, dtype=str)
    cols = {c.lower(): c for c in df.columns}
    cve_col = cols.get("cve") or cols.get("cve_id")
    epss_col = cols.get("epss")
    pct_col = cols.get("percentile")
    date_col = cols.get("date") or cols.get("created") or cols.get("epss_date")
    if not (cve_col and epss_col and pct_col):
        raise RuntimeError("EPSS CSV must have columns: cve (or cve_id), epss, percentile")
    out = pd.DataFrame({
        "cve_id": df[cve_col].astype(str).map(normalize_cve_id),
        "epss_score": pd.to_numeric(df[epss_col], errors="coerce"),
        "epss_percentile": pd.to_numeric(df[pct_col], errors="coerce"),
        "epss_date": df[date_col] if date_col else pd.NA
    })
    return out.drop_duplicates(subset=["cve_id"])

# ---------- priority ----------
def compute_priority(row) -> str:
    """
    Updated triage:
      - Act Now: KEV
      - High: EPSS pct >= 0.90
      - Medium: (EPSS pct >= 0.50) OR (severity in {CRITICAL,HIGH} AND cvss_vector suggests remote & low-complexity)
      - Low: else
    """
    sev = str(row.get("severity","")).upper()
    kev = str(row.get("in_cisa_kev","")).lower() == "true"
    epss = row.get("epss_score", float("nan"))
    pct  = row.get("epss_percentile", float("nan"))
    vector = str(row.get("cvss_vector","")).upper()

    if kev:
        return "Act Now"
    if pd.notna(pct) and pct >= 0.90:
        return "High"
    remote_easy = ("AV:N" in vector) and ("AC:L" in vector)
    if (pd.notna(pct) and pct >= 0.50) or (sev in ("CRITICAL","HIGH") and remote_easy):
        return "Medium"
    return "Low"

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Enrich CPPS CVEs with KEV and EPSS")
    ap.add_argument("--kev_csv",  help="Local path to KEV CSV (offline mode)")
    ap.add_argument("--epss_csv", help="Local path to EPSS CSV (offline mode)")
    args = ap.parse_args()

    df = load_input(INPUT_CSV)
    print(f"[LOAD] Input rows: {len(df)}")

    # KEV
    try:
        kev = load_kev(args.kev_csv) if args.kev_csv else load_kev(None)
        print(f"[KEV] Loaded {len(kev)} KEV entries")
        df = df.merge(kev, how="left", on="cve_id")
    except Exception as e:
        print(f"[KEV] ERROR: {e}. Continuing without KEV.", file=sys.stderr)
        df["in_cisa_kev"] = False
        df["kev_added_dt"] = pd.NA

    # EPSS
    try:
        if args.epss_csv:
            epss = load_epss_from_csv(args.epss_csv)
            print(f"[EPSS] Loaded offline EPSS rows: {len(epss)}")
        else:
            unique_cves = df["cve_id"].dropna().unique().tolist()
            epss = fetch_epss_recursive(unique_cves, max_batch=150)
            print(f"[EPSS] Retrieved EPSS rows: {len(epss)}")
        df = df.merge(epss, how="left", on="cve_id")
    except Exception as e:
        print(f"[EPSS] ERROR: {e}. Continuing without EPSS.", file=sys.stderr)
        df["epss_score"] = pd.NA
        df["epss_percentile"] = pd.NA
        df["epss_date"] = pd.NA

    # Priority



    ################################################################################################################################
    # --- ICSA merge (insert right after the EPSS block, before "# Priority") ---
        # --- ICSA merge (right after the EPSS block, before "# Priority") ---
    try:
        icsa_path = Path("icsa_advisories.csv")  # adjust if you keep a different folder
        if not icsa_path.exists():
            print(f"[ICSA] WARN: {icsa_path} not found. Continuing without ICSA.", file=sys.stderr)
            # keep columns present so downstream code doesn't explode
            for col, default in (
                ("icsa_present", 0),
                ("icsa_ids", pd.NA),
                ("icsa_urls", pd.NA),
                ("icsa_cwes", pd.NA),
                ("icsa_cvss_v3", pd.NA),
                ("icsa_first_seen", pd.NA),
            ):
                if col not in df.columns:
                    df[col] = default
        else:
            icsa = pd.read_csv(icsa_path, dtype=str)

            # many-to-one: one advisory can list many CVEs; aggregate advisory metadata per CVE
            agg = (
                icsa.groupby("cve", dropna=False)
                    .agg(
                        icsa_present=("icsa_id", lambda s: int(s.notna().any())),
                        icsa_ids=("icsa_id", lambda s: ";".join(sorted(set([x for x in s if pd.notna(x)])))),
                        icsa_urls=("icsa_url", lambda s: ";".join(sorted(set([x for x in s if pd.notna(x)])))),
                        icsa_cwes=("icsa_cwes", lambda s: ";".join(sorted(set(";".join([x for x in s if pd.notna(x)]).split(";"))))),
                        icsa_cvss_v3=("icsa_cvss_v3", "max"),
                        icsa_first_seen=("icsa_pub_date", "min"),
                    )
                    .reset_index()
                    .rename(columns={"cve": "cve_id"})
            )

            df = df.merge(agg, on="cve_id", how="left")
            df["icsa_present"] = df["icsa_present"].fillna(0).astype(int)

            merged_icsa = int(df["icsa_present"].sum())
            print(f"[ICSA] Merged advisory data for {merged_icsa} CVEs")
    except Exception as e:
        print(f"[ICSA] WARN: {e}. Continuing without ICSA.", file=sys.stderr)
        # keep columns present so downstream code doesn't explode
        if "icsa_present" not in df.columns:
            df["icsa_present"] = 0
        for col in ("icsa_ids","icsa_urls","icsa_cwes","icsa_cvss_v3","icsa_first_seen"):
            if col not in df.columns:
                df[col] = pd.NA
    # --- end ICSA merge ---


    ##################################################################################################################################
    df["priority"] = df.apply(compute_priority, axis=1)

    # Debug prints
    matched_kev = (df["in_cisa_kev"] == True).sum() if "in_cisa_kev" in df else 0
    matched_epss = df["epss_score"].notna().sum()
    print(f"[RESULT] KEV matches: {matched_kev} | EPSS matches: {matched_epss}")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[SAVE] {len(df)} rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

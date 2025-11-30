#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import pandas as pd, numpy as np, yaml

def load_configs(mapping_path, config_yaml, hazard_path):
    mappings = pd.read_csv(mapping_path, comment='#')
    with open(config_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    hazards = pd.read_csv(hazard_path)
    return mappings, cfg, hazards

RENAME = {
    "cve":"cve_id","cveid":"cve_id","cve-id":"cve_id",
    "vendor_name":"vendor","product_name":"product",
    "cvss":"cvss_score","cvssv3":"cvss_score","cvss_v3_score":"cvss_score","cvss_v31_score":"cvss_score",
    "vector":"cvss_vector","cvss_vector_string":"cvss_vector",
    "is_kev":"in_cisa_kev","kev":"in_cisa_kev",
    "epss":"epss_percentile",
    "published_date":"published","publishedat":"published",
    "installed_version":"installed_version","version":"installed_version"
}

def normalize_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})

    # existing defaults...
    for c, default in [
        ("vendor",""), ("product",""),
        ("cvss_score", np.nan), ("cvss_vector",""),
        ("epss_percentile", np.nan), ("in_cisa_kev", 0),
        ("published", None),
        ("installed_version", None), ("affected_range", None), ("cpe", None),

        # NEW: ICSA/CERT fields
        ("icsa_present", 0),
        ("icsa_ids", pd.NA),
        ("icsa_urls", pd.NA),
        ("icsa_cwes", pd.NA),
        ("icsa_cvss_v3", pd.NA),
        ("icsa_first_seen", pd.NA),
    ]:
        if c not in df.columns:
            df[c] = default

    # If ICSA provides a CVSS but NVD is missing, borrow it
    if "icsa_cvss_v3" in df.columns:
        df["cvss_score"] = df["cvss_score"].fillna(pd.to_numeric(df["icsa_cvss_v3"], errors="coerce"))

    # Parse ICSA date if present
    if "icsa_first_seen" in df.columns:
        df["icsa_first_seen"] = pd.to_datetime(df["icsa_first_seen"], errors="coerce")

    # Ensure published is datetime for later logic
    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")

    return df


def map_asset_type_row(vendor, product, mappings):
    import pandas as _pd
    v = "" if (vendor is None or (isinstance(vendor,float) and _pd.isna(vendor))) else str(vendor)
    p = "" if (product is None or (isinstance(product,float) and _pd.isna(product))) else str(product)
    for _, r in mappings.iterrows():
        try:
            if re.search(r["vendor_regex"], v, flags=re.IGNORECASE) and re.search(r["product_regex"], p, flags=re.IGNORECASE):
                return r["asset_type"], r["function"], int(r["criticality_0_5"]), int(r["exposure_0_5"])
        except re.error:
            continue
    return "Other ICS Device", "Industrial device", 3, 3

def likelihood(cvss, epss_pct, kev_flag, vector, weights):
    cvss_norm = np.nan if pd.isna(cvss) else max(0.0, min(float(cvss)/10.0, 1.0))
    epss_norm = 0.25 if pd.isna(epss_pct) else max(0.0, min(float(epss_pct), 1.0))
    kev = 1.0 if bool(kev_flag) else 0.0
    L = weights["cvss"] * (0.5 if pd.isna(cvss_norm) else cvss_norm) + weights["epss"] * epss_norm + weights["kev"] * kev
    vec = str(vector or "").upper()
    if "AV:N" in vec and "AC:L" in vec:
        L += weights.get("avn_acl_boost", 0.0)
    return min(L, 1.0)

def impact_from_priors(crit_0_5, expo_0_5):
    return 0.7*(crit_0_5/5.0) + 0.3*(expo_0_5/5.0)

def env_factor(row, hazards, env_weights):
    loc = row.get("location", None)
    if loc is not None and "location" in hazards.columns:
        match = hazards[hazards["location"].astype(str).str.lower()==str(loc).lower()]
        h = match.iloc[0] if not match.empty else hazards.iloc[0]
    else:
        h = hazards.iloc[0]
    e = (env_weights["heat_weight"]*float(h.get("heat_0_5",0)) + env_weights["flood_weight"]*float(h.get("flood_0_5",0)) + env_weights["storm_weight"]*float(h.get("storm_0_5",0))) / env_weights.get("scale",5.0)
    return min(max(e,0.0),1.0)

def version_status(installed_version, affected_range):
    if not installed_version or not isinstance(installed_version, str):
        return "unknown"
    if not affected_range or not isinstance(affected_range, str) or affected_range.strip()=="" or affected_range.lower()=="unknown":
        return "unknown"
    try:
        from packaging import version
    except Exception:
        return "unknown"
    iv = version.parse(installed_version.strip())
    clauses = [c.strip() for c in re.split(r"[,\s]+", affected_range) if c.strip()]
    ok = True; any_clause=False
    for c in clauses:
        any_clause=True
        m = re.match(r"(<=|>=|<|>|==|=)?\s*([0-9a-zA-Z\.\-\+]+)", c)
        if not m: continue
        op, ver = m.groups()
        try: tv = version.parse(ver)
        except Exception: continue
        if op in (None,"==","="): ok = ok and (iv == tv)
        elif op == "<": ok = ok and (iv < tv)
        elif op == "<=": ok = ok and (iv <= tv)
        elif op == ">": ok = ok and (iv > tv)
        elif op == ">=": ok = ok and (iv >= tv)
    if not any_clause: return "unknown"
    return "match" if ok else "mismatch"

def explain_row(row):
    reasons = []
    if int(row.get("icsa_present", 0)) == 1:
        reasons.append("ICSA")
    if bool(row.get("in_cisa_kev", False)):
        reasons.append("KEV")
    vec = str(row.get("cvss_vector","")).upper()
    if "AV:N" in vec and "AC:L" in vec:
        reasons.append("AV:N+AC:L")
    if float(row.get("impact_I",0)) >= 0.8:
        reasons.append("HighCriticality")
    if float(row.get("impact_I",0)) >= 0.6 and float(row.get("E",0)) >= 0.3:
        reasons.append("HighImpactWithEnv")
    if float(row.get("epss_percentile",0)) >= 0.5:
        reasons.append("HighEPSS")
    vm = row.get("version_match","unknown")
    if vm == "unknown":
        reasons.append("VersionUnknown")
    elif vm == "mismatch":
        reasons.append("VersionMismatchDiscount")
    return ",".join(reasons) if reasons else ""


def compute_priority_quantiles(risks, quantiles_cfg):
    qs = {k: float(v) for k, v in quantiles_cfg.items()}
    ordered = sorted(qs.items(), key=lambda kv: kv[1], reverse=True)
    cuts = {label: risks.quantile(q) for label, q in ordered}
    def assign(r):
        for label, qv in ordered:
            if r >= cuts[label]:
                return label
        return "Low"
    return risks.apply(assign)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", default="power_grid_cves_enriched.csv")
    ap.add_argument("-o","--output", default="data/all_cves_risk_v2.csv")
    ap.add_argument("--mappings", default="config/mappings.csv")
    ap.add_argument("--config", default="config/risk_config.yaml")
    ap.add_argument("--hazards", default="data/hazard_table.csv")
    ap.add_argument("--site", default="Nashville")
    args = ap.parse_args()

    in_path = Path(args.input); out_path = Path(args.output)
    mappings, cfg, hazards = load_configs(Path(args.mappings), Path(args.config), Path(args.hazards))
    weights = cfg["risk_weights"]; qcfg = cfg["priority_bands_quantiles"]; discounts = cfg["discounts"]; envw = cfg["env_weights"]

    df = pd.read_csv(in_path)
    df = normalize_cols(df)

    mapped = df[["vendor","product","cvss_score","cvss_vector","epss_percentile","in_cisa_kev","published"]].copy()
    atypes=[]; funcs=[]; crits=[]; expos=[]
    for v,p in zip(df["vendor"], df["product"]):
        at, fn, cr, ex = map_asset_type_row(v, p, mappings)
        atypes.append(at); funcs.append(fn); crits.append(cr); expos.append(ex)
    mapped["asset_type"]=atypes; mapped["function"]=funcs; mapped["criticality_0_5"]=crits; mapped["exposure_0_5"]=expos

    if "location" in df.columns:
        mapped["location"] = df["location"].fillna(args.site).replace("", args.site)
    else:
        mapped["location"] = args.site

    mapped["L"] = [likelihood(cv, ep, kev, vec, weights) for cv,ep,kev,vec in zip(mapped["cvss_score"], mapped["epss_percentile"], mapped["in_cisa_kev"], mapped["cvss_vector"])]
    # --- ICSA-driven tweak to Likelihood ---
    if "icsa_present" in df.columns:
        mapped["icsa_present"] = df["icsa_present"].fillna(0).astype(int)
        icsa_bump = float(weights.get("icsa_like_bump", 0.05))  # default +0.05
        # Optional recency decay. Off by default unless you set a half-life in config.
        hl_days = weights.get("icsa_recency_half_life_days", None)

        if hl_days:
            # decays from 1.0 to 0.5 over 'hl_days', then continues halving each half-life
            def decay(d):
                if pd.isna(d): return 1.0
                delta = (pd.Timestamp.utcnow() - d.tz_localize("UTC") if d.tzinfo else pd.Timestamp.utcnow() - d).days
                return 0.5 ** max(delta, 0) / float(hl_days)
            bump_vec = mapped.get("icsa_first_seen")
            # If first_seen missing, treat as full bump
            bump_scale = bump_vec.apply(decay) if bump_vec is not None else 1.0
            mapped.loc[mapped["icsa_present"] == 1, "L"] = (
                mapped.loc[mapped["icsa_present"] == 1, "L"] + icsa_bump * bump_scale
            ).clip(upper=1.0)
        else:
            mapped.loc[mapped["icsa_present"] == 1, "L"] = (
                mapped.loc[mapped["icsa_present"] == 1, "L"] + icsa_bump
            ).clip(upper=1.0)
    # --- end ICSA tweak ---

    mapped["impact_I"] = [impact_from_priors(c,e) for c,e in zip(mapped["criticality_0_5"], mapped["exposure_0_5"])]
    mapped["E"] = [env_factor(row, hazards, envw) for _, row in mapped.iterrows()]


    vs=[]
    for iv, ar in zip(df.get("installed_version", [None]*len(df)), df.get("affected_range", [None]*len(df))):
        vs.append(version_status(iv, ar))
    mapped["version_match"] = vs

    env_mult = weights.get("env_multiplier", 0.3)
    mapped["Risk_base"] = mapped["L"] * (mapped["impact_I"] * (1 + env_mult * mapped["E"]))

    disc=[]
    for vm in mapped["version_match"]:
        if vm=="unknown": disc.append(discounts.get("version_unknown", 1.0))
        elif vm=="mismatch": disc.append(discounts.get("version_mismatch", 1.0))
        else: disc.append(1.0)
    mapped["Risk"] = (mapped["Risk_base"] * pd.Series(disc)).clip(upper=1.0)

    mapped["priority"] = compute_priority_quantiles(mapped["Risk"], qcfg)
    mapped["explain"] = [explain_row(r) for _, r in mapped.iterrows()]

    # Attach ids back
    for col in ["cve_id","vendor","product","severity","epss_percentile","in_cisa_kev","cvss_score","cvss_vector","published"]:
        if col not in mapped.columns and col in df.columns:
            mapped[col] = df[col]

    cols = [
    "cve_id","vendor","product","asset_type","function",
    "cvss_score","cvss_vector","epss_percentile","in_cisa_kev",
    "icsa_present","icsa_ids","icsa_urls","icsa_cwes","icsa_cvss_v3","icsa_first_seen",
    "criticality_0_5","exposure_0_5","location",
    "L","impact_I","E","Risk_base","version_match","Risk","priority",
    "explain","published"
    ]
    keep = [c for c in cols if c in mapped.columns]
    out = mapped[keep]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    roll = out.groupby(["vendor","product","asset_type"], dropna=False)["Risk"].max().reset_index()
    roll.to_csv(out_path.parent / "product_risk_rollup_v2.csv", index=False)

    print(f"Wrote {out_path} and {out_path.parent/'product_risk_rollup_v2.csv'}")

if __name__ == "__main__":
    main()

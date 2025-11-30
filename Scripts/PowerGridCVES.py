# Severity_PowerGridCVES.py
# Extract CPPS/Smart-Grid–related CVEs from NVD JSON feeds (2020–2025)
# - No CLI args; filenames are hardcoded (zip/gz/json are supported)
# - Handles nested configurations and both 'criteria'/'cpe23Uri'
# - Prefers vulnerable CPEs; falls back to any CPE if none are vulnerable
# - Extracts CVSS v3.1 → v3.0 → v2 (v2 severity from baseSeverity OR severity)
# - Writes explicit v31/v2 fields + an overall (cvss_score, cvss_vector, severity)
# - Adds matching_keywords + audit columns for vulnerable CPEs
# - Robust CSV saving on Windows (Excel lock) with timestamp fallback

from __future__ import annotations

import os
import re
import io
import json
import gzip
import zipfile
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Set

import pandas as pd

# -------------------------------------------------------------------
# Feeds (use whichever extension is present)
# -------------------------------------------------------------------
DATA_STEMS = [
    "nvdcve-2.0-2020",
    "nvdcve-2.0-2021",
    "nvdcve-2.0-2022",
    "nvdcve-2.0-2023",
    "nvdcve-2.0-2024",
    "nvdcve-2.0-2025",
]
POSSIBLE_EXTS = [".json.zip", ".json.gz", ".json"]

OUTPUT_CSV = "power_grid_cves.csv"

# -------------------------------------------------------------------
# Keyword and regex sets (expanded for CPPS/Smart Grid)
# -------------------------------------------------------------------

POWER_KEYWORDS = [
    # systems & roles
    "substation", "substation automation", "sas", "scada", "ems", "dms", "oms", "hmi", "engineering workstation",
    # devices
    "ied", "intelligent electronic device", "protection relay", "recloser controller", "feeder automation",
    "rtu", "plc", "gateway", "protocol converter", "phasor measurement unit", "pmu",
    "phasor data concentrator", "pdc", "synchrophasor",
    # protocols & standards
    "iec 61850", "sampled values", "mms", "dnp3", "opendnp3", "modbus", "modbus/tcp", "opc ua",
    "iec 60870-5-104", "iec 60870-5-101", "tase.2", "iccp", "ieee c37.118", "dlms", "cosem",
    # smart grid & ami
    "smart grid", "ami", "amr", "smart meter", "meter data management", "mdm", "head-end system", "hes",
    # der & storage
    "der", "derms", "microgrid", "bess", "battery energy storage", "inverter", "power conversion system", "pcs",
    "hvdc", "statcom", "svc", "facts",
    # vendors/products (common)
    "siemens", "ruggedcom", "spectrum power", "siprotec", "sicam",
    "abb", "hitachi energy", "microscada", "relion",
    "ge grid", "ge digital", "ifix", "cimplicity", "e-terra", "eterra",
    "schneider electric", "modicon", "micom", "ecostruxure",
    "schweitzer engineering",
    "emerson", "yokogawa", "honeywell", "rockwell automation", "allen-bradley", "omron",
    "mitsubishi electric", "beckhoff", "wago", "ignition", "inductive automation",
    "moxa", "hirschmann", "belden", "cisco industrial ethernet",
    # keep earlier tokens
    "power grid", "switchgear", "cyber-physical", "opal-rt",
]

# List of (label, compiled_regex)
REGEX_KEYWORDS: List[Tuple[str, re.Pattern]] = [
    ("iec61850-goose-a",  re.compile(r"\biec\s*61850\b.*\bgoose\b", re.IGNORECASE)),
    ("iec61850-goose-b",  re.compile(r"\bgoose\b.*\biec\s*61850\b", re.IGNORECASE)),
    ("iec61850-sampled-values", re.compile(r"\biec\s*61850\b.*\bsampled\s*values?\b", re.IGNORECASE)),
    ("iccp-tase2",        re.compile(r"\btase\.?2\b|\biccp\b", re.IGNORECASE)),
    ("iec60870-5-10x",    re.compile(r"\biec\s*60870[-\s]?5[-/]10[14]\b", re.IGNORECASE)),
    ("sel-family-num",    re.compile(r"\bsel[-\s]?\d{3,4}\b", re.IGNORECASE)),
    ("sel-family-other",  re.compile(r"\bsel[-\s]?(rtac|3530|axion)\b", re.IGNORECASE)),
    ("micom",             re.compile(r"\bmi?com\b", re.IGNORECASE)),
    ("relion",            re.compile(r"\brelion\b", re.IGNORECASE)),
    ("siprotec",          re.compile(r"\bsiprotec\b", re.IGNORECASE)),
    ("sicam",             re.compile(r"\bsicam\b", re.IGNORECASE)),
    ("microscada",        re.compile(r"\bmicro\s*scada\b|\bmicroscada\b", re.IGNORECASE)),
    ("spectrum-power",    re.compile(r"\bspectrum\s*power\b", re.IGNORECASE)),
    ("eterra",            re.compile(r"\be-?terra\b|\beterra\b", re.IGNORECASE)),
    ("ecostruxure",       re.compile(r"\becostruxure\b", re.IGNORECASE)),
    ("ami-vendors",       re.compile(r"\bitron\b|\blandis\+?gyr\b|\bsensus\b|\belster\b|\baklara\b|\bkamstrup\b", re.IGNORECASE)),
]

# -------------------------------------------------------------------
# Utilities: reading feeds
# -------------------------------------------------------------------
def read_feed(path: str) -> Dict[str, Any]:
    if path.endswith(".json.zip"):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".json")]
            if not names:
                names = zf.namelist()
            with zf.open(names[0]) as f:
                data = f.read()
                return json.loads(data.decode("utf-8", errors="replace"))
    elif path.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported feed type: {path}")

def find_existing_feed(stem: str) -> Optional[str]:
    for ext in POSSIBLE_EXTS:
        p = f"{stem}{ext}"
        if os.path.exists(p):
            return p
    return None

# -------------------------------------------------------------------
# CPE helpers
# -------------------------------------------------------------------
def extract_vendor_product_from_cpe23(cpe23: str) -> Tuple[Optional[str], Optional[str]]:
    parts = cpe23.split(":")
    vendor = parts[3] if len(parts) > 4 else None
    product = parts[4] if len(parts) > 5 else None
    if vendor:
        vendor = vendor.replace("-", " ")
    if product:
        product = product.replace("-", " ")
    return vendor, product

def iter_nodes(conf: Any) -> Iterator[Dict[str, Any]]:
    """Yield all nodes (depth-first) from configurations in either dict/list shapes."""
    if isinstance(conf, dict):
        nodes = conf.get("nodes")
        if isinstance(nodes, list):
            for n in nodes:
                yield n
                for child in iter_nodes(n.get("children", [])):
                    yield child
        elif conf.get("cpeMatch") or conf.get("children"):
            yield conf
            for child in iter_nodes(conf.get("children", [])):
                yield child
    elif isinstance(conf, list):
        for item in conf:
            yield from iter_nodes(item)

def iter_cpe_matches(conf_section: Any) -> Iterator[Dict[str, Any]]:
    for node in iter_nodes(conf_section):
        cml = node.get("cpeMatch", [])
        if isinstance(cml, list):
            for cm in cml:
                yield cm

def select_vendor_product(conf_section: Any) -> Tuple[Optional[str], Optional[str], List[str], int]:
    vulnerable_uris: List[str] = []
    any_uris: List[str] = []

    for cm in iter_cpe_matches(conf_section):
        uri = cm.get("criteria") or cm.get("cpe23Uri")
        if not uri:
            continue
        if cm.get("vulnerable", False):
            vulnerable_uris.append(uri)
        any_uris.append(uri)

    chosen_uri = None
    if vulnerable_uris:
        chosen_uri = vulnerable_uris[0]
    elif any_uris:
        chosen_uri = any_uris[0]

    vendor = product = None
    if chosen_uri:
        vendor, product = extract_vendor_product_from_cpe23(chosen_uri)

    return vendor, product, vulnerable_uris, len(vulnerable_uris)

# -------------------------------------------------------------------
# CVSS helpers
# -------------------------------------------------------------------
def _pick_metric(metric_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metric_list:
        return None
    prim = [m for m in metric_list if str(m.get("type", "")).lower() == "primary"]
    return prim[0] if prim else metric_list[0]

def _extract_v31(metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    lst = metrics.get("cvssMetricV31")
    if not isinstance(lst, list):
        return None, None, None
    m = _pick_metric(lst)
    if not m:
        return None, None, None
    data = m.get("cvssData", {}) or {}
    score = data.get("baseScore")
    vector = data.get("vectorString")
    sev = m.get("baseSeverity") or m.get("severity")
    return score, vector, sev

def _extract_v30(metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    lst = metrics.get("cvssMetricV30")
    if not isinstance(lst, list):
        return None, None, None
    m = _pick_metric(lst)
    if not m:
        return None, None, None
    data = m.get("cvssData", {}) or {}
    score = data.get("baseScore")
    vector = data.get("vectorString")
    sev = m.get("baseSeverity") or m.get("severity")
    return score, vector, sev

def _extract_v2(metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    lst = metrics.get("cvssMetricV2")
    if not isinstance(lst, list):
        return None, None, None
    m = _pick_metric(lst)
    if not m:
        return None, None, None
    data = m.get("cvssData", {}) or {}
    score = data.get("baseScore")
    vector = data.get("vectorString")
    sev = m.get("baseSeverity") or m.get("severity")
    return score, vector, sev

def extract_cvss(metrics: Dict[str, Any]) -> Dict[str, Tuple[Optional[float], Optional[str], Optional[str]]]:
    v31 = _extract_v31(metrics)
    v30 = _extract_v30(metrics)
    v2 = _extract_v2(metrics)
    return {"v31": v31, "v30": v30, "v2": v2}

def choose_overall(cv: Dict[str, Tuple[Optional[float], Optional[str], Optional[str]]]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    for key in ("v31", "v30", "v2"):
        s, v, se = cv.get(key, (None, None, None))
        if s is not None:
            if not se:
                se = cv.get(key, (None, None, None))[2]
            return s, v, se
    return None, None, None

# -------------------------------------------------------------------
# Matching helpers
# -------------------------------------------------------------------
def keyword_matches(text: str, keywords: Iterable[str]) -> Set[str]:
    hits: Set[str] = set()
    txt = (text or "").lower()
    for kw in keywords:
        k = kw.lower()
        if any(c in k for c in (" ", "-", "/")):
            if k in txt:
                hits.add(kw)
        else:
            if re.search(r"\b" + re.escape(k) + r"\b", txt):
                hits.add(kw)
    return hits

def regex_matches(text: str, regex_list: List[Tuple[str, re.Pattern]]) -> Set[str]:
    """Return set of regex labels that matched in text, given a list of (label, compiled_pattern)."""
    hits: Set[str] = set()
    t = text or ""
    for label, pattern in regex_list:
        if pattern.search(t):
            hits.add(label)
    return hits

# -------------------------------------------------------------------
# Extraction per feed
# -------------------------------------------------------------------
def extract_from_feed(feed: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    vulns = feed.get("vulnerabilities", [])
    for entry in vulns:
        cve = entry.get("cve", {})
        cve_id = cve.get("id")
        published = cve.get("published")

        # Description (English)
        desc = ""
        for d in cve.get("descriptions", []) or []:
            if d.get("lang") == "en":
                desc = d.get("value") or ""
                break

        # CVSS
        metrics = cve.get("metrics", {}) or {}
        cv = extract_cvss(metrics)
        v31_score, v31_vec, v31_sev = cv["v31"]
        v2_score,  v2_vec,  v2_sev  = cv["v2"]

        overall_score, overall_vec, overall_sev = choose_overall(cv)
        if not overall_sev:
            overall_sev = v31_sev or v2_sev

        # CWE
        cwe_id = None
        weaknesses = cve.get("weaknesses", []) or []
        if weaknesses:
            wdesc = weaknesses[0].get("description", []) or []
            if wdesc:
                cwe_id = wdesc[0].get("value")

        # References
        refs = [r.get("url") for r in cve.get("references", []) or [] if r.get("url")]
        ref_joined = "; ".join(refs)

        # Configurations (CPEs)
        configurations = cve.get("configurations")
        vendor = product = None
        vulnerable_cpes: List[str] = []
        vuln_cpe_count = 0
        if configurations is not None:
            vendor, product, vulnerable_cpes, vuln_cpe_count = select_vendor_product(configurations)

        # Matching keywords (why included)
        fields_for_match = " ".join([part for part in [desc, vendor or "", product or ""] if part])
        hits_plain = keyword_matches(fields_for_match, POWER_KEYWORDS)
        hits_regex = regex_matches(fields_for_match, REGEX_KEYWORDS)
        matched = sorted(hits_plain.union(hits_regex))

        if not matched:
            continue

        out.append(
            {
                "cve_id": cve_id,
                "published": published,
                "vendor": vendor,
                "product": product,
                # explicit versioned fields
                "cvss_v31_base_score": v31_score,
                "cvss_v31_base_severity": v31_sev,
                "cvss_v2_base_score": v2_score,
                "cvss_v2_severity": v2_sev,
                # overall
                "cvss_score": overall_score,
                "cvss_vector": overall_vec,
                "severity": (overall_sev or None),
                "cwe_id": cwe_id,
                "description": desc,
                "references": ref_joined,
                # audits
                "vulnerable_cpes": "; ".join(vulnerable_cpes) if vulnerable_cpes else None,
                "vuln_cpe_count": vuln_cpe_count,
                "matching_keywords": "; ".join(matched),
            }
        )
    return out

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main() -> None:
    all_rows: List[Dict[str, Any]] = []

    for stem in DATA_STEMS:
        path = find_existing_feed(stem)
        if not path:
            print(f"[INFO] Feed not found for {stem} (tried {POSSIBLE_EXTS})")
            continue
        try:
            feed = read_feed(path)
            rows = extract_from_feed(feed)
            all_rows.extend(rows)
            print(f"[OK] {os.path.basename(path)} → {len(rows)} matching CVEs")
        except Exception as e:
            print(f"[WARN] Failed to process {path}: {e}")

    # Dedup by CVE ID (last wins)
    dedup: Dict[str, Dict[str, Any]] = {}
    for r in all_rows:
        if not r.get("cve_id"):
            continue
        dedup[r["cve_id"]] = r

    df = pd.DataFrame(list(dedup.values()))
    if not df.empty and "published" in df.columns:
        df = df.sort_values("published", ascending=False)

    # Robust save (Windows/Excel lock safe)
    out_path = OUTPUT_CSV
    def try_save(path: str) -> bool:
        try:
            df.to_csv(path, index=False, encoding="utf-8")
            return True
        except PermissionError:
            return False

    if not try_save(out_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = f"power_grid_cves_{ts}.csv"
        if not try_save(alt):
            docs = os.path.join(os.path.expanduser("~"), "Documents")
            os.makedirs(docs, exist_ok=True)
            final = os.path.join(docs, alt)
            df.to_csv(final, index=False, encoding="utf-8")
            print(f"[SAVE] CSV locked; saved to {final}")
        else:
            print(f"[SAVE] CSV locked; saved to {alt}")
    else:
        print(f"[SAVE] Saved {len(df)} records to {out_path}")

if __name__ == "__main__":
    main()

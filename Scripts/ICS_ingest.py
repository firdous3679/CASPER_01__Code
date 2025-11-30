# icsa_ingest.py
import re, time, csv, sys
from pathlib import Path
from urllib.parse import urljoin
import requests
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ICSA-Ingest/1.0; +https://example.org)"
}

FEEDS = [
    "https://www.cisa.gov/cybersecurity-advisories/ics-advisories.xml",
    "https://www.cisa.gov/cybersecurity-advisories/ics-medical-advisories.xml",
]

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.I)
CWE_RE = re.compile(r"CWE-\d{1,5}", re.I)
ICSA_RE = re.compile(r"ICSA-\d{2}-\d{3}(?:-\d{2})?", re.I)
CVSS_RE = re.compile(r"CVSS\s*v?3[\.\d]*\s*(?:Base\s*Score\s*[:=]?\s*)?([0-9]\.[0-9])", re.I)

def fetch(url, expect_xml=False, retries=3, sleep=1.0):
    for i in range(retries):
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.text
        time.sleep(sleep * (i + 1))
    raise RuntimeError(f"Failed to fetch {url} ({r.status_code})")

def parse_rss(xml_text):
    root = ET.fromstring(xml_text)
    # works for most RSS/Atom variants CISA uses
    ns = {"atom": "http://www.w3.org/2005/Atom", "rss": "http://purl.org/rss/1.0/"}  # best effort
    items = []
    # try Atom
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        link = link_el.attrib.get("href") if link_el is not None else None
        title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        pub = (entry.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
        if link:
            items.append({"title": title, "link": link, "pubDate": pub})
    # try RSS
    if not items:
        for entry in root.findall(".//item"):
            link = (entry.findtext("link") or "").strip()
            title = (entry.findtext("title") or "").strip()
            pub = (entry.findtext("pubDate") or "").strip()
            if link:
                items.append({"title": title, "link": link, "pubDate": pub})
    return items

def parse_advisory(html, url):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    # extract fields
    icsa_id = None
    # try heading text, then anywhere in page
    h1 = soup.find("h1")
    if h1:
        m = ICSA_RE.search(h1.get_text(" ", strip=True))
        if m: icsa_id = m.group(0).upper()
    if not icsa_id:
        m = ICSA_RE.search(text)
        if m: icsa_id = m.group(0).upper()
    # date
    pub_date = None
    time_el = soup.find("time")
    if time_el and time_el.get("datetime"):
        pub_date = time_el["datetime"]
    # cves
    cves = sorted(set(CVE_RE.findall(text)))
    # cwes
    cwes = sorted(set(CWE_RE.findall(text)))
    # cvss
    cvss = None
    m = CVSS_RE.search(text)
    if m:
        cvss = m.group(1)
    # vendor/products: look for "Affected Products" section
    vendor_products = None
    for h in soup.find_all(["h2","h3","h4"]):
        if "affected product" in h.get_text(" ", strip=True).lower():
            blk = []
            # collect siblings until next heading
            for sib in h.find_all_next():
                if sib.name in ["h2","h3","h4"]:
                    break
                blk.append(sib.get_text(" ", strip=True))
            vendor_products = " | ".join([s for s in blk if s]).strip()[:4000]
            break
    if not vendor_products:
        # fallback: a lot of content is already in text; grab a slice around "Affected Products"
        i = text.lower().find("affected product")
        if i != -1:
            vendor_products = text[i:i+1500]
    return {
        "icsa_id": icsa_id,
        "icsa_url": url,
        "icsa_pub_date": pub_date,
        "icsa_cvss_v3": cvss,
        "icsa_cwes": ";".join(cwes) if cwes else None,
        "icsa_vendor_products": vendor_products
    }, cves

def collect_icsa():
    items = []
    for feed in FEEDS:
        try:
            xml = fetch(feed, expect_xml=True)
            items.extend(parse_rss(xml))
        except Exception as e:
            print(f"[warn] feed failed {feed}: {e}", file=sys.stderr)
    # dedupe by link
    seen = set()
    out_rows = []
    for it in items:
        link = it["link"]
        if link in seen: 
            continue
        seen.add(link)
        try:
            html = fetch(link)
            meta, cves = parse_advisory(html, link)
            if not cves:
                # advisory without explicit CVEs still useful for context; keep a placeholder
                out_rows.append({**meta, "cve": None, "title": it["title"], "icsa_feed_date": it["pubDate"]})
            else:
                for c in cves:
                    out_rows.append({**meta, "cve": c.upper(), "title": it["title"], "icsa_feed_date": it["pubDate"]})
        except Exception as e:
            print(f"[warn] advisory failed {link}: {e}", file=sys.stderr)
            continue
    return out_rows

def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("[info] no rows to write"); return
    cols = ["cve","icsa_id","title","icsa_pub_date","icsa_cvss_v3","icsa_cwes","icsa_vendor_products","icsa_url","icsa_feed_date"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})

if __name__ == "__main__":
    rows = collect_icsa()
    write_csv(rows, "data/icsa_advisories.csv")
    print(f"[done] wrote {len(rows)} rows to data/icsa_advisories.csv")

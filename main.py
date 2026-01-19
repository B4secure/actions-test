import os
import re
import glob
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta
import shutil

import feedparser
import pandas as pd
import numpy as np
from dateutil import parser as dateparser

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# CONFIG (can be overridden by GitHub Actions env vars)
# ---------------------------
PAST_DAYS = int(os.getenv("PAST_DAYS", "1"))
LOOKBACK_HOURS = int((os.getenv("LOOKBACK_HOURS") or "18").strip())
MAX_ITEMS = int(os.getenv("MAX_ITEMS", "30"))
DUP_THRESHOLD = float(os.getenv("DUP_THRESHOLD", "0.60"))
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

DEFAULT_HL, DEFAULT_GL, DEFAULT_CEID = "en-GB", "GB", "GB:en"


REGION_RULES = [
    # Belgium
    (r"Belgium.*Dutch",  ("nl-BE", "BE", "BE:nl")),
    (r"Belgium.*French", ("fr-BE", "BE", "BE:fr")),
    (r"Belgium.*English",("en-GB", "BE", "BE:en")),

    # Germany
    (r"Germany.*German", ("de-DE", "DE", "DE:de")),
    (r"Germany.*English",("en-GB", "DE", "DE:en")),

    # Spain
    (r"Spain.*Spanish",  ("es-ES", "ES", "ES:es")),
    (r"Spain.*English",  ("en-GB", "ES", "ES:en")),

    # France
    (r"France.*French",  ("fr-FR", "FR", "FR:fr")),
    (r"France.*English", ("en-GB", "FR", "FR:en")),

    # Italy
    (r"Italy.*Italian",  ("it-IT", "IT", "IT:it")),
    (r"Italy.*English",  ("en-GB", "IT", "IT:en")),
]


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------
# Your search library (unchanged)
# ---------------------------
SEARCH_LIBRARY_TEXT = r"""
Brand search Google News and X - PROTEST   ("Gucci" OR "Ralph Lauren" OR "Dior" OR "Armani" OR "Louis Vuitton" OR "Valentino" OR "Burberry" OR "Dolce Gabbana" OR "Polo Ralph Lauren" OR "Cartier" OR "Calvin Klein" OR "Tommy Hilfiger" OR "Timberland" OR "Prada" OR "Nike" OR "Balenciaga" OR "Canada Goose") AND protest
Brand search Google News and X - BOYCOTT   ("Gucci" OR "Ralph Lauren" OR "Dior" OR "Armani" OR "Louis Vuitton" OR "Valentino" OR "Burberry" OR "Dolce Gabbana" OR "Polo Ralph Lauren" OR "Cartier" OR "Calvin Klein" OR "Tommy Hilfiger" OR "Timberland" OR "Prada" OR "Nike" OR "Balenciaga" OR "Canada Goose") AND boycott
Cotswold Designer Outlet   ("Cotswold Designer Outlet")
Value Retail	"Value Retail" OR "Bicester Collection"
London Marylebone   ("London Marylebone" OR "Marylebone station") AND (incident OR disruption OR delay OR closure OR police OR protest OR bomb OR explosion OR stabbing OR shooting OR "suspicious package")
Roermond outlet   ("Roermond Outlet" OR Roermond) AND (incident OR protest OR boycott OR police OR closure OR evacuation OR bomb OR explosion OR stabbing OR shooting OR crime)
BV Logistics Companies   ("DHL" OR "UPS" OR "DPD" OR "FedEx" OR "Amazon" OR "Royal Mail" OR "Parcelforce" OR "EVRI") AND (delivery OR supply OR logistics)
High Level City searches Belgium Dutch language   ("Brussels" OR "Maastricht" OR "Antwerp") AND (bom OR bommelding OR explosie OR schieten OR steekincident OR "verdacht pakket")
High Level City searches Belgium English language   ("Brussels" OR "Maastricht" OR "Antwerp") AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package" OR "bomb threat")
High Level City searches Belgium French language   ("Brussels" OR "Maastricht" OR "Antwerp") AND (bombe OR explosion OR "colis suspect" OR poignarder OR fusillade)
Local Town Searches Maasmechelen Dutch	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (incident OR protest OR boycot OR bom OR bommelding OR explosie OR evacuatie OR afzetting OR politie OR politie-inzet OR steekpartij OR steekincident OR schietpartij OR schieten OR dreiging OR "verdacht pakket" OR verdachte)
Local Town Searches Maasmechelen French	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (incident OR manifestation OR protestation OR bombe OR "alerte à la bombe" OR "menace de bombe" OR explosion OR évacuation OR "colis suspect" OR agression OR "attaque au couteau" OR fusillade OR poignarder OR police OR menace)
Local Town Searches Maasmechelen English	("Maasmechelen" OR "Hasselt" OR "Lanaken") AND (incident OR protest OR boycott OR police OR "armed police" OR evacuation OR cordon OR closure OR disruption OR delay OR threat OR terror OR bomb OR explosion OR shooting OR stabbing OR "suspicious package" OR "bomb threat")
Village searches Maasmechelen   ("Maasmechelen Village" OR ("Maasmechelen" AND ("designer outlet" OR outlet OR retail)))
High Level City searches Germany English   ("Frankfurt" OR "Cologne" OR "Munich") AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package")
High Level City searches Germany German   ("Frankfurt" OR "Cologne" OR "Munich") AND (bombe OR explosion OR schießen OR messer OR "verdächtiges paket")
Local Town Searches Wertheim English	("Wertheim" OR "Wurzburg" OR "Aschaffenburg") AND (incident OR protest OR boycott OR police OR "armed police" OR evacuation OR cordon OR closure OR disruption OR threat OR bomb OR explosion OR shooting OR stabbing OR "suspicious package" OR "bomb threat")
Local Town Searches Wertheim German	("Wertheim" OR "Wurzburg" OR "Aschaffenburg") AND (incident OR protest OR boykott OR polizei OR polizeieinsatz OR bombe OR bombendrohung OR sprengsatz OR explosion OR evakuierung OR absperrung OR verdächtig OR "verdächtiges paket" OR schießen OR schuss OR messer OR messerangriff OR großalarm OR großeinsatz)
Village search Wertheim   ("Wertheim Village" OR ("Wertheim" AND ("designer outlet" OR outlet OR retail)))
XR and JSO Broad search   ("Extinction Rebellion" OR "Just Stop Oil")
XR and JSO Village search   ("Extinction Rebellion" OR "Just Stop Oil") AND ("Ingolstadt" OR "Kildare" OR "Bicester" OR "Wertheim" OR "Fidenza" OR "Maasmechelen")
Shoplifting UK   "shoplifting" OR "retail crime" OR "theft from person" OR "pickpocket gang"
High Level City searches Spain English   ("Madrid" OR "Barcelona") AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package")
High Level City searches Spain Spanish   ("Madrid" OR "Barcelona") AND (bomba OR explosión OR tiroteo OR puñalada OR "paquete sospechoso")
Local Town Searches Las Rozas & La Roca English   ("Las Rozas" OR "La Roca" OR "Mataro" OR "Badalona") AND (protest OR boycott OR bomb OR explosion OR shooting OR stabbing)
Local Town Searches Las Rozas & La Roca Spanish   ("Las Rozas" OR "La Roca" OR "Mataro" OR "Badalona") AND (protesta OR boicot OR bomba OR explosión OR tiroteo OR puñalada)
Village Search Las Rozas & La Roca   ("Las Rozas Village" OR i"La Roca Village")
Local Town Searches Bicester & Kildare	("Bicester" OR "Kildare" OR "Newbridge") AND (protest OR boycott OR bomb OR explosion OR shooting OR stabbing)
Village Search Bicester & Kildare   ("Bicester Village" OR "Kildare Village")
High Level City searches UK & Ireland   ("London" OR "Oxford" OR "Dublin") AND (bomb OR explosion OR shooting OR stabbing OR "suspicious package")
High Level City searches France English	"Paris" AND (bomb OR explosion OR shooting OR stabbing)
High Level City searches France French	"Paris" AND (bombe OR explosion OR fusillade OR poignarder)
Local Town Searches La Vallée French   ("Serris" OR "Chessy" OR "Disneyland Paris") AND (manifestation OR grève OR bombe OR explosion OR poignarder)
Village Search La Vallée	("La Vallée Village" OR ("Serris" OR "Chessy") AND ("designer outlet" OR outlet OR retail))
High Level City searches Italy English   ("Milan" OR "Bologna") AND (bomb OR explosion OR shooting OR stabbing)
High Level City searches Italy Italian   ("Milan" OR "Bologna") AND (bomba OR esplosione OR sparatoria OR accoltellamento)
Local Town Searches Fidenza Italian   ("Fidenza" OR "Parma" OR "Piacenza" OR "Cremona") AND (protesta OR boicottaggio OR bomba OR esplosione)
Village Search Fidenza	("Fidenza Village" OR ("Fidenza" AND ("designer outlet" OR outlet OR retail)))


""".strip()



def parse_published_dt(published_str: str):
    if not published_str:
        return None
    try:
        dt = dateparser.parse(published_str)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def filter_last_n_hours(df, hours: int):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    df = df.copy()
    df["published_dt_utc"] = df["published"].apply(parse_published_dt)
    df = df[df["published_dt_utc"].notna()]
    df = df[df["published_dt_utc"] >= cutoff].reset_index(drop=True)
    return df

def edition_for_search(search_name: str) -> tuple[str, str, str]:
    """
    Decide (hl, gl, ceid) for a given search based on search_name.
    Falls back to UK edition if nothing matches.
    """
    name = (search_name or "").strip()
    for pattern, triple in REGION_RULES:
        if re.search(pattern, name, flags=re.IGNORECASE):
            return triple
    return (DEFAULT_HL, DEFAULT_GL, DEFAULT_CEID)


def parse_search_library(text: str) -> pd.DataFrame:
    rows = []
    pending_name = None  # used when we detect "Name<TAB>" and then URLs follow on next lines

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # If it's a URL on its own line, treat as a search entry (best effort)
        if line.startswith("http"):
            if pending_name:
                rows.append({"search_name": pending_name, "raw_query": f'"{line}"'})
            else:
                rows.append({"search_name": "URL_SOURCE", "raw_query": f'"{line}"'})
            continue

        # Preferred: tab-separated "name<TAB>query"
        if "\t" in line:
            name, query = line.split("\t", 1)
            name = name.strip()
            query = query.strip()

            # If query is empty, the next lines might be URLs (multi-line block)
            if query == "":
                pending_name = name
                continue

            # If query contains embedded newlines (rare in a single 'line'), split and keep valid URLs too
            # (This mainly protects against accidental pasted blocks.)
            if "\n" in query:
                parts = [p.strip().strip('"') for p in query.splitlines() if p.strip()]
                if parts and parts[0].startswith("http"):
                    for u in parts:
                        rows.append({"search_name": name, "raw_query": f'"{u}"'})
                    pending_name = None
                    continue

            rows.append({"search_name": name, "raw_query": query})
            pending_name = None
            continue

        # Fallback: split on 2+ spaces (covers many "Name    query" lines)
        m = re.split(r"\s{2,}", line, maxsplit=1)
        if len(m) == 2:
            name, query = m[0].strip(), m[1].strip()
            rows.append({"search_name": name, "raw_query": query})
            pending_name = None
            continue

        # Otherwise can't parse safely
        rows.append({"search_name": "UNMAPPED_LINE", "raw_query": line})

    return pd.DataFrame(rows)


def is_google_news_compatible(q: str) -> bool:
    q = (q or "").strip().lower()
    if not q:
        return False
    if q.startswith("http://") or q.startswith("https://"):
        return False
    if "to:" in q or q.startswith("@") or " @" in q:
        return False
    return True


def google_news_rss_url(query: str, past_days: int, hl: str, gl: str, ceid: str) -> str:
    full = f"{query} when:{past_days}d"
    q = urllib.parse.quote(full)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"



def collect_google_news(df_searches: pd.DataFrame, past_days: int, max_items: int) -> pd.DataFrame:
    out_rows = []

    for _, r in df_searches.iterrows():
        name = r["search_name"]
        q = r["raw_query"]

        # Choose the right Google News edition for this query
        hl, gl, ceid = edition_for_search(name)

        rss = google_news_rss_url(q, past_days, hl=hl, gl=gl, ceid=ceid)
        feed = feedparser.parse(rss)

        for entry in feed.entries[:max_items]:
            out_rows.append(
                {
                    "search_name": name,
                    "search_query": q,
                    "title": entry.get("title", ""),
                    "published": entry.get("published", ""),
                    "link": entry.get("link", ""),
                    "past_days": past_days,
                    "hl": hl,
                    "gl": gl,
                    "ceid": ceid,
                }
            )

    return pd.DataFrame(out_rows)



def latest_file(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return files[0]


def semantic_dedupe_csv(infile: str, out_clean: str, out_audit: str,
                        threshold: float, model_name: str) -> tuple[int, int]:
    df = pd.read_excel(infile)
    df["compare_text"] = df["title"].fillna("").astype(str)

    mask = df["compare_text"].str.len() > 0
    df_work = df[mask].copy().reset_index(drop=True)
    orig_idx = df.index[mask].to_numpy()

    if df_work.empty:
        df.drop(columns=["compare_text"], errors="ignore").to_excel(out_clean, index=False)
        pd.DataFrame().to_excel(out_audit, index=False, engine="openpyxl")
        return len(df), len(df)

    model = SentenceTransformer(model_name)
    emb = model.encode(
        df_work["compare_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    sim = cosine_similarity(emb, emb)
    n = sim.shape[0]

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    keep_work = set()
    audit_rows = []

    for g in groups.values():
        if len(g) == 1:
            keep_work.add(g[0])
            continue

        g_map = [(int(orig_idx[i]), i) for i in g]
        g_map.sort(key=lambda x: x[0])

        keep_orig, keep_i = g_map[0]
        keep_work.add(keep_i)

        for drop_orig, drop_i in g_map[1:]:
            audit_rows.append(
                {
                    "kept_original_row": keep_orig,
                    "dropped_original_row": int(drop_orig),
                    "similarity": float(sim[keep_i, drop_i]),
                    "kept_title": df.loc[keep_orig, "title"],
                    "dropped_title": df.loc[int(drop_orig), "title"],
                }
            )

    kept_orig_rows = {int(orig_idx[i]) for i in keep_work}
    drop_orig_rows = set(map(int, orig_idx.tolist())) - kept_orig_rows

    keep_mask = np.ones(len(df), dtype=bool)
    for r in drop_orig_rows:
        keep_mask[r] = False

    df_clean = df.loc[keep_mask].drop(columns=["compare_text"], errors="ignore").reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)

    df_clean.to_excel(out_clean, index=False, engine = "openpyxl")
    audit.to_excel(out_audit, index=False, engine = "openpyxl")
    return len(df), len(df_clean)


def main():
    ts = datetime.now(timezone.utc).strftime("%d_%m%y_UTC")

    search_df = parse_search_library(SEARCH_LIBRARY_TEXT)
    to_run = search_df.copy()


    results = collect_google_news(to_run, past_days=PAST_DAYS, max_items=MAX_ITEMS)
    results = filter_last_n_hours(results, hours=LOOKBACK_HOURS)

    

    if not results.empty:
        results = results.drop_duplicates(subset=["link"]).reset_index(drop=True)

    raw_results_file = DATA_DIR / f"google_news_raw_{ts}_past{PAST_DAYS}d.xlsx"
    audit_search_file = DATA_DIR / f"search_audit_{ts}.xlsx"

    
    results = results.apply(lambda s: s.dt.tz_localize(None) if hasattr(s, "dt") and getattr(s.dt, "tz", None) is not None else s)

    results.to_excel(raw_results_file, index=False, engine = "openpyxl")
    search_df.to_excel(audit_search_file, index=False, engine = "openpyxl")

    skipped = search_df[search_df["search_name"] == "UNMAPPED_LINE"]
    print(f"Parsed {len(to_run)} runnable searches; skipped {len(skipped)} unmapped lines")
    if not skipped.empty:
        print(skipped.head(20).to_string(index=False))


    # Dedupe the raw file we just created
    dedup_file = DATA_DIR / f"google_news_dedup_{ts}_past{PAST_DAYS}d.xlsx"
    dedup_audit = DATA_DIR / f"google_news_dedup_audit_{ts}.xlsx"

    orig, cleaned = semantic_dedupe_csv(
        infile=str(raw_results_file),
        out_clean=str(dedup_file),
        out_audit=str(dedup_audit),
        threshold=DUP_THRESHOLD,
        model_name=MODEL_NAME,
    )
    # Always keep a stable single file for automation
    latest = DATA_DIR / "latest_deduped.xlsx"
    shutil.copyfile(dedup_file, latest)
    print(f"Saved latest: {latest}")
    print("Total searches to run:", len(to_run))
    print(to_run["search_name"].value_counts().head(30))



    print(f"Saved raw:   {raw_results_file} | rows={len(results)}")
    print(f"Saved audit: {audit_search_file} | searches={len(search_df)}")
    print(f"Dedupe: original={orig} cleaned={cleaned}")
    print(f"Saved dedup: {dedup_file}")
    print(f"Saved dedup audit: {dedup_audit}")
    print(f"Running with LOOKBACK_HOURS={LOOKBACK_HOURS}")


if __name__ == "__main__":
    main()


# %%













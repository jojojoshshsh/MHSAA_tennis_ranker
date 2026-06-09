import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

src_dir = Path("src/rankings_by_division_flight")
out_dir = Path("docs")
out_dir.mkdir(exist_ok=True)

csv_dir = out_dir / "csv"
csv_dir.mkdir(exist_ok=True)

ALLOWED_FLIGHTS = {"1", "2", "3"}
# ── Team rankings ────────────────────────────────────────────
team_data = []
for csv_path in sorted(src_dir.glob("team_*.csv")):
    df = pd.read_csv(csv_path)
    if df.empty:
        continue
    stem = csv_path.stem  # e.g. team_boys_division_1
    gender   = "Boys"  if "_boys_"  in stem else "Girls"
    division = stem.split("_division_")[-1].replace("_", " ")
    team_data.append({
        "gender":   gender,
        "division": division,
        "df":       df.head(10),
    })

DIVISION_ORDER_T = {"1": 0, "2": 1, "3": 2, "4": 3, "4 other": 4}
team_data.sort(key=lambda x: (
    DIVISION_ORDER_T.get(x["division"], 9),
    0 if x["gender"] == "Boys" else 1,
))

# Build team rankings HTML
team_html = ""
for entry in team_data:
    label  = f"Top 10 Teams · {entry['gender']} Division {entry['division']}"
    anchor = f"team_{entry['gender'].lower()}_div{entry['division'].replace(' ','')}"
    df     = entry["df"]

    thead = "<thead><tr>" + "".join(
        f'<th onclick="sortTable(this)">{col}</th>'
        for col in df.columns
    ) + "</tr></thead>"
    tbody = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{row[col]}</td>" for col in df.columns) + "</tr>"
        for _, row in df.iterrows()
    ) + "</tbody>"

    team_html += f"""
    <section id="{anchor}">
      <div class="section-header">
        <h2>{label}</h2>
      </div>
      <div class="table-wrap">
        <table class="rankings-table">{thead}{tbody}</table>
      </div>
    </section>
    """

# Load all CSVs and filter
all_data = []
for csv_path in sorted(src_dir.glob("*.csv")):
    df = pd.read_csv(csv_path)
    if "flight" in df.columns:
        df = df[df["flight"].astype(str).isin(ALLOWED_FLIGHTS)]
    if df.empty:
        continue

    dest = csv_dir / csv_path.name
    df.to_csv(dest, index=False)

    # Parse division, flight, category (singles/doubles), gender from filename
    # filename pattern: singles_boys_division_1_flight_1.csv
    stem = csv_path.stem
    category = "singles" if stem.startswith("singles") else "doubles"
    gender   = "boys"    if "_boys_"    in stem else "girls"

    # Extract division and flight from the dataframe itself (more reliable)
    if "division" in df.columns and "flight" in df.columns:
        for (division, flight), group in df.groupby(["division", "flight"]):
            all_data.append({
                "division": str(division),
                "flight":   str(flight),
                "category": category,
                "gender":   gender,
                "filename": csv_path.name,
                "df":       group.copy(),
            })

# Sort: division -> flight -> gender -> category
DIVISION_ORDER = {"1": 0, "2": 1, "3": 2, "4": 3, "4_other": 4}
GENDER_ORDER   = {"boys": 0, "girls": 1}
CAT_ORDER      = {"singles": 0, "doubles": 1}

all_data.sort(key=lambda x: (
    DIVISION_ORDER.get(x["division"], 9),
    x["flight"],
    GENDER_ORDER.get(x["gender"], 9),
    CAT_ORDER.get(x["category"], 9),
))

# Build nav groups: division -> list of (label, anchor)
from collections import defaultdict
nav_groups = defaultdict(list)

tables_html = ""
for entry in all_data:
    division = entry["division"]
    flight   = entry["flight"]
    category = entry["category"].title()
    gender   = entry["gender"].title()
    filename = entry["filename"]
    df       = entry["df"]

    preview_cols = [c for c in [
        "rank", "name", "pair_name", "school",
        "division", "flight", "wins", "losses",
        "TGRS", "TGRS_scaled", "ts_rating", "ts_mu", "local_ts_mu", "ts_sigma",
        "reachability", "local_reachability",
        "sos", "local_sos", "quality_wins",
        "last_match_date"
    ] if c in df.columns]

    anchor = f"div{division}_flight{flight}_{gender.lower()}_{category.lower()}"
    label  = f"Div {division} · Flight {flight} · {gender} {category}"

    thead = "<thead><tr>" + "".join(
        f'<th onclick="sortTable(this)">{col}</th>'
        for col in preview_cols
    ) + "</tr></thead>"

    tbody = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{row[col]}</td>" for col in preview_cols) + "</tr>"
        for _, row in df.head(30).iterrows()
    ) + "</tbody>"

    tables_html += f"""
    <section id="{anchor}">
      <div class="section-header">
        <h2>{label}</h2>
        <a class="dl-btn" href="csv/{filename}">Download CSV</a>
      </div>
      <div class="table-wrap">
        <table class="rankings-table">
          {thead}
          {tbody}
        </table>
      </div>
    </section>
    """

    nav_groups[f"Division {division}"].append((label, anchor))

# Build nav HTML grouped by division
team_nav = "".join(
    f'<a href="#team_{e["gender"].lower()}_div{e["division"].replace(" ","")}">Teams · {e["gender"]} D{e["division"]}</a>'
    for e in team_data
)
nav_html = ""
for div_label, links in nav_groups.items():
    nav_html += f'<span class="nav-group-label">{div_label}</span>'
    nav_html += "".join(f'<a href="#{anchor}">{lbl}</a>' for lbl, anchor in links)

edt = timezone(timedelta(hours=-4))
updated = datetime.now(edt).strftime("%B %d, %Y at %I:%M %p EDT")
section_count = len(all_data)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Michigan High School Tennis Rankings</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f7fa; color: #1a1a2e; line-height: 1.5; }}
  header {{ background: #1a3a5c; color: white; padding: 2rem 1.5rem 1.5rem; }}
  header h1 {{ font-size: 1.6rem; font-weight: 600; margin-bottom: .4rem; }}
  header p {{ opacity: .8; font-size: .9rem; }}
  nav {{
    background: #132d47; padding: .75rem 1.5rem;
    display: flex; flex-wrap: wrap; gap: .4rem; align-items: center;
  }}
  .nav-group-label {{
    color: #4a90c4; font-size: .7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: .05em;
    padding: .2rem .5rem .2rem 0;
    margin-left: .5rem;
  }}
  .nav-group-label:first-child {{ margin-left: 0; }}
  nav a {{
    color: #b8d8f0; text-decoration: none; font-size: .78rem;
    padding: .2rem .5rem; border-radius: 4px;
    border: 1px solid rgba(255,255,255,.1);
  }}
  nav a:hover {{ background: rgba(255,255,255,.12); }}
  main {{ max-width: 1400px; margin: auto; padding: 1.5rem; }}
  section {{
    background: white; border-radius: 10px; padding: 1.25rem;
    margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.07);
  }}
  .section-header {{
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 1rem;
  }}
  h2 {{ font-size: 1.05rem; font-weight: 600; color: #1a3a5c; }}
  .dl-btn {{
    font-size: .8rem; color: #1a3a5c; text-decoration: none;
    border: 1px solid #c0d4e8; border-radius: 6px; padding: .3rem .7rem;
  }}
  .dl-btn:hover {{ background: #e8f0f8; }}
  .table-wrap {{ overflow-x: auto; }}
  .rankings-table {{ width: 100%; border-collapse: collapse; font-size: .78rem; white-space: nowrap; }}
  .rankings-table th {{
    background: #1a3a5c; color: white; padding: 6px 10px;
    text-align: left; font-weight: 500;
    cursor: pointer; user-select: none;
  }}
  .rankings-table th:hover {{ background: #245180; }}
  .rankings-table th.asc::after  {{ content: " ▲"; font-size: .65rem; }}
  .rankings-table th.desc::after {{ content: " ▼"; font-size: .65rem; }}
  .rankings-table td {{ padding: 5px 10px; border-bottom: 1px solid #eef0f3; }}
  .rankings-table tr:nth-child(even) td {{ background: #f8fafc; }}
  .rankings-table tr:hover td {{ background: #eef4fb; }}
  .rankings-table td:first-child {{ font-weight: 600; color: #1a3a5c; width: 36px; }}
  footer {{ text-align: center; color: #888; font-size: .78rem; padding: 2rem; }}
  .nav-about {{ color: #ffd580; font-size: .8rem; padding: .2rem .6rem; border-radius: 4px; border: 1px solid rgba(255,213,128,.3); text-decoration: none; margin-right: .75rem; }}
  .nav-about:hover {{ background: rgba(255,213,128,.1); }}
</style>
</head>
<body>
<header>
  <h1>Michigan High School Tennis Rankings</h1>
  <p>Updated automatically everyday at 4am EDT. Last update: {updated}.</p>
</header>
<nav><a class="nav-about" href="about.html">About &amp; Methodology</a>{team_nav}<span class="nav-group-label">Individual</span>{nav_html}</nav>
<main>
{team_html}
{tables_html}
</main>
<footer>Rankings computed using TrueSkill + Graph Reachability (TGRS). Data from TennisReporting.com.</footer>
</body>
<script>
function sortTable(th) {{
  const tbody = th.closest('table').querySelector('tbody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  const col   = Array.from(th.parentElement.children).indexOf(th);
  const asc   = !th.classList.contains('asc');
  th.closest('thead').querySelectorAll('th').forEach(h => h.classList.remove('asc','desc'));
  th.classList.add(asc ? 'asc' : 'desc');
  rows.sort((a, b) => {{
    const av = a.cells[col].textContent.trim();
    const bv = b.cells[col].textContent.trim();
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</html>"""

(out_dir / "index.html").write_text(html, encoding="utf-8")
print(f"Built docs/index.html with {section_count} sections")

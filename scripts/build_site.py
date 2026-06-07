import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

src_dir = Path("src/rankings_by_division_flight")
out_dir = Path("docs")
out_dir.mkdir(exist_ok=True)

csv_dir = out_dir / "csv"
csv_dir.mkdir(exist_ok=True)

ALLOWED_FLIGHTS = {"1", "2", "3"}

sections = {}
for csv_path in sorted(src_dir.glob("*.csv")):
    df = pd.read_csv(csv_path)
    if "flight" in df.columns:
        df = df[df["flight"].astype(str).isin(ALLOWED_FLIGHTS)]
    if df.empty:
        continue
    dest = csv_dir / csv_path.name
    df.to_csv(dest, index=False)
    label = csv_path.stem.replace("_", " ").title()
    sections[label] = (csv_path.name, df)

tables_html = ""
for label, (filename, df) in sections.items():
    preview_cols = [c for c in [
        "rank", "name", "pair_name", "school",
        "division", "flight", "wins", "losses",
        "TGRS", "ts_rating", "ts_mu", "local_ts_mu", "ts_sigma",
        "reachability", "local_reachability",
        "sos", "local_sos", "quality_wins",
        "last_match_date"
    ] if c in df.columns]

    # Build table header manually so we can add sort onclick
    thead = "<thead><tr>" + "".join(
        f'<th onclick="sortTable(this)">{col}</th>'
        for col in preview_cols
    ) + "</tr></thead>"

    # Build table body
    tbody = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{row[col]}</td>" for col in preview_cols) + "</tr>"
        for _, row in df.head(30).iterrows()
    ) + "</tbody>"

    table_id = filename.replace(".csv", "").replace(".", "_")

    tables_html += f"""
    <section id="{filename.replace('.csv','')}">
      <div class="section-header">
        <h2>{label}</h2>
        <a class="dl-btn" href="csv/{filename}">Download CSV</a>
      </div>
      <div class="table-wrap">
        <table class="rankings-table" id="tbl_{table_id}">
          {thead}
          {tbody}
        </table>
      </div>
    </section>
    """

# EDT = UTC-4
edt = timezone(timedelta(hours=-4))
updated = datetime.now(edt).strftime("%B %d, %Y at %I:%M %p EDT")
csv_count = len(sections)

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
  nav {{ background: #132d47; padding: .6rem 1.5rem; display: flex; flex-wrap: wrap; gap: .5rem; }}
  nav a {{ color: #7fb8e8; text-decoration: none; font-size: .8rem; padding: .2rem .5rem; border-radius: 4px; }}
  nav a:hover {{ background: rgba(255,255,255,.1); }}
  main {{ max-width: 1400px; margin: auto; padding: 1.5rem; }}
  section {{ background: white; border-radius: 10px; padding: 1.25rem; margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.07); }}
  .section-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }}
  h2 {{ font-size: 1.05rem; font-weight: 600; color: #1a3a5c; }}
  .dl-btn {{ font-size: .8rem; color: #1a3a5c; text-decoration: none; border: 1px solid #c0d4e8; border-radius: 6px; padding: .3rem .7rem; }}
  .dl-btn:hover {{ background: #e8f0f8; }}
  .table-wrap {{ overflow-x: auto; }}
  .rankings-table {{ width: 100%; border-collapse: collapse; font-size: .78rem; white-space: nowrap; }}
  .rankings-table th {{
    background: #1a3a5c; color: white; padding: 6px 10px;
    text-align: left; font-weight: 500;
    cursor: pointer; user-select: none; position: relative;
  }}
  .rankings-table th:hover {{ background: #245180; }}
  .rankings-table th.asc::after  {{ content: " ▲"; font-size: .65rem; }}
  .rankings-table th.desc::after {{ content: " ▼"; font-size: .65rem; }}
  .rankings-table td {{ padding: 5px 10px; border-bottom: 1px solid #eef0f3; }}
  .rankings-table tr:nth-child(even) td {{ background: #f8fafc; }}
  .rankings-table tr:hover td {{ background: #eef4fb; }}
  .rankings-table td:first-child {{ font-weight: 600; color: #1a3a5c; width: 36px; }}
  footer {{ text-align: center; color: #888; font-size: .78rem; padding: 2rem; }}
</style>
</head>
<body>
<header>
  <h1>Michigan High School Tennis Rankings</h1>
  <p>Updated automatically every Monday at 6am EDT. Last update: {updated}. {csv_count} divisions.</p>
</header>
<nav>
  {"".join(f'<a href="#{fn.replace(".csv","")}">{lbl}</a>' for lbl, (fn, _) in sections.items())}
</nav>
<main>
{tables_html}
</main>
<footer>Rankings computed using TrueSkill + Graph Reachability (TGRS). Data from TennisReporting.com.</footer>
</body>
<script>
function sortTable(th) {{
  const table = th.closest('table');
  const tbody = table.querySelector('tbody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  const col   = Array.from(th.parentRow || th.parentElement.children).indexOf(th);
  const asc   = !th.classList.contains('asc');

  // Clear all headers in this table
  th.closest('thead').querySelectorAll('th').forEach(h => h.classList.remove('asc','desc'));
  th.classList.add(asc ? 'asc' : 'desc');

  rows.sort((a, b) => {{
    const av = a.cells[col].textContent.trim();
    const bv = b.cells[col].textContent.trim();
    const an = parseFloat(av);
    const bn = parseFloat(bv);
    const numericSort = !isNaN(an) && !isNaN(bn);
    if (numericSort) return asc ? an - bn : bn - an;
    return asc ? av.localeCompare(bv) : bv.localeCompare(av);
  }});

  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</html>"""

(out_dir / "index.html").write_text(html, encoding="utf-8")
print(f"Built docs/index.html with {csv_count} tables")

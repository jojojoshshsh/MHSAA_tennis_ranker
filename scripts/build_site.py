function getBestPerFlight(school) {{
  school = school.trim().toLowerCase();
  const result = {{}};
  for (const [stem, data] of Object.entries(CSV_DATA)) {{
    if (stem.startsWith('team_')) continue;
    const cols      = data.cols;
    const schoolIdx = cols.indexOf('school');
    const rankIdx   = cols.indexOf('rank');
    if (schoolIdx === -1) continue;

    const category = stem.startsWith('singles') ? 'Singles' : 'Doubles';
    const gender   = stem.includes('_boys_') ? 'Boys' : 'Girls';
    const divIdx    = cols.indexOf('division');
    const flightIdx = cols.indexOf('flight');

    for (const row of data.rows) {{
      if (!String(row[schoolIdx]).toLowerCase().includes(school)) continue;
      const div    = divIdx    >= 0 ? row[divIdx]    : '?';
      const flight = flightIdx >= 0 ? row[flightIdx] : '?';
      const rank   = rankIdx   >= 0 ? Number(row[rankIdx]) : 9999;
      const key = `${{gender}} ${{category}} · Div ${{div}} · Flight ${{flight}}`;
      if (!result[key] || rank < result[key].rank) {{
        result[key] = {{ rank, cols, row }};
      }}
    }}
  }}
  return result;
}}

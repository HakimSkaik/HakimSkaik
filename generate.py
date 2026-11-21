"""
Builds dark_mode.svg and light_mode.svg — a neofetch-style profile card.

Run locally:     python generate.py              (stats show as "--")
Run with stats:  GH_TOKEN=... python generate.py (the GitHub Action does this daily)
"""
import os
import time
from datetime import date
from html import escape

import requests

USER = "HakimSkaik"

# ── Edit your card here ─────────────────────────────────────────────
# A tuple is (label, value). None draws a blank line.
# ("§", "Title") draws a section header.
# BIRTHDAY: set to date(YYYY, M, D) to show an "Uptime" line, or leave None.
BIRTHDAY = None

INFO = [
    ("OS", "macOS, Linux, Windows"),
    "UPTIME",
    ("Host", "Labelbox"),
    ("Kernel", "Applied AI Engineer"),
    ("Startup", "Heart.Fi — Co-founder & CEO"),
    ("IDE", "Claude Code, VS Code"),
    None,
    ("Languages.Programming", "Python, TypeScript, Java, C"),
    ("Languages.Real", "Arabic, English, French"),
    ("Stack.AI", "LangGraph, LangChain, MCP, PyTorch"),
    ("Stack.Web", "React, Node.js, Web3"),
    ("Education", "Concordia SWE '26"),
    None,
    ("§", "Contact"),
    ("Website", "hakim-skaik.dev"),
    ("GitHub", "github.com/HakimSkaik"),
    None,
    ("§", "GitHub Stats"),
    "STATS",
]
# ────────────────────────────────────────────────────────────────────

WIDTH_CH = 58        # characters per info line
FONT = 15            # px
CH = FONT * 0.6      # monospace glyph width
LH = 20              # line height
ART_FONT = 11        # portrait is drawn smaller for more detail
ART_LH = 13
ART_X = 24
INFO_X = 462
TOP = 32

THEMES = {
    "dark": dict(bg="#161b22", border="#30363d", text="#c9d1d9", key="#ffa657",
                 head="#58a6ff", dots="#616e7f", add="#3fb950", dele="#f85149",
                 art="#adbac7", art_file="assets/ascii_dark.txt"),
    "light": dict(bg="#f6f8fa", border="#d0d7de", text="#24292f", key="#953800",
                  head="#0969da", dots="#afb8c1", add="#1a7f37", dele="#cf222e",
                  art="#424a53", art_file="assets/ascii_light.txt"),
}


# ── GitHub stats ────────────────────────────────────────────────────
def gql(token, query, variables=None):
    r = requests.post("https://api.github.com/graphql",
                      json={"query": query, "variables": variables or {}},
                      headers={"Authorization": f"bearer {token}"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]


def fetch_stats(token):
    d = gql(token, """
    query($login:String!){ user(login:$login){
      createdAt
      followers{ totalCount }
      contributionsCollection{ contributionYears }
      repositoriesContributedTo(contributionTypes:[COMMIT,PULL_REQUEST,REPOSITORY]){ totalCount }
    }}""", {"login": USER})["user"]

    # Owned repos + stars (paginated)
    repos, stars, cursor = [], 0, None
    while True:
        page = gql(token, """
        query($login:String!,$after:String){ user(login:$login){
          repositories(first:100, after:$after, ownerAffiliations:OWNER, isFork:false){
            totalCount pageInfo{ hasNextPage endCursor }
            nodes{ nameWithOwner stargazerCount }
        }}}""", {"login": USER, "after": cursor})["user"]["repositories"]
        repos += [n["nameWithOwner"] for n in page["nodes"]]
        stars += sum(n["stargazerCount"] for n in page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        cursor = page["pageInfo"]["endCursor"]

    # Total commits across every contribution year
    commits = 0
    for y in d["contributionsCollection"]["contributionYears"]:
        c = gql(token, """
        query($login:String!,$from:DateTime!,$to:DateTime!){ user(login:$login){
          contributionsCollection(from:$from,to:$to){
            totalCommitContributions restrictedContributionsCount }}}""",
                {"login": USER, "from": f"{y}-01-01T00:00:00Z",
                 "to": f"{y}-12-31T23:59:59Z"})["user"]["contributionsCollection"]
        commits += c["totalCommitContributions"] + c["restrictedContributionsCount"]

    # Lines added/removed by you in your own repos
    add = rem = 0
    hdr = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    for repo in repos:
        for _ in range(6):  # GitHub returns 202 while it computes stats
            r = requests.get(f"https://api.github.com/repos/{repo}/stats/contributors",
                             headers=hdr, timeout=30)
            if r.status_code != 202:
                break
            time.sleep(3)
        if r.status_code != 200 or not isinstance(r.json(), list):
            continue
        for c in r.json():
            if c.get("author") and c["author"]["login"].lower() == USER.lower():
                add += sum(w["a"] for w in c["weeks"])
                rem += sum(w["d"] for w in c["weeks"])

    return dict(repos=len(repos), contrib=d["repositoriesContributedTo"]["totalCount"],
                stars=stars, commits=commits, followers=d["followers"]["totalCount"],
                add=add, rem=rem)


# ── SVG building ────────────────────────────────────────────────────
def fmt(n):
    return "--" if n is None else f"{n:,}"


def uptime():
    if not BIRTHDAY:
        return None
    t = date.today()
    months = (t.year - BIRTHDAY.year) * 12 + (t.month - BIRTHDAY.month)
    if t.day < BIRTHDAY.day:
        months -= 1
    y, m = divmod(months, 12)
    # date of the most recent "month-iversary"
    ay, am = divmod(BIRTHDAY.year * 12 + BIRTHDAY.month - 1 + months, 12)
    import calendar
    anchor = date(ay, am + 1, min(BIRTHDAY.day, calendar.monthrange(ay, am + 1)[1]))
    return f"{y} years, {m} months, {(t - anchor).days} days"


def kv_line(label, value, width=WIDTH_CH):
    """label: ........ value  (right-aligned to width)"""
    dots = width - len(label) - 2 - len(value) - 2
    return [("key", label), ("dots", ": " + "." * max(dots, 1) + " "), ("text", value)]


def split_line(left, right, width=WIDTH_CH):
    """Two key/values on one line: 'Repos: .. 15 {Contributed: 3} | Stars: .... 12'"""
    half = width // 2 + 4
    l = kv_line(*left, width=half - 2)
    r = kv_line(*right, width=width - half - 1)
    return l + [("dots", " | ")] + r


def build_info(stats):
    rows = []
    rows.append([("head", f"{USER.lower()}@github "), ("dots", "─" * (WIDTH_CH - len(USER) - 8))])
    for item in INFO:
        if item is None:
            rows.append([])
        elif item == "UPTIME":
            u = uptime()
            if u:
                rows.append(kv_line("Uptime", u))
        elif item == "STATS":
            s = stats or {}
            repos = f"{fmt(s.get('repos'))} {{Contributed: {fmt(s.get('contrib'))}}}"
            rows.append(split_line(("Repos", repos), ("Stars", fmt(s.get("stars")))))
            rows.append(split_line(("Commits", fmt(s.get("commits"))),
                                   ("Followers", fmt(s.get("followers")))))
            loc = None if not stats else s["add"] - s["rem"]
            seg = [("key", "Lines of Code"), ("dots", ": "), ("text", fmt(loc)),
                   ("dots", " ( "), ("add", fmt(s.get("add")) + "++"), ("dots", ", "),
                   ("dele", fmt(s.get("rem")) + "--"), ("dots", " )")]
            rows.append(seg)
        elif item[0] == "§":
            title = item[1]
            rows.append([("head", "— " + title + " "),
                         ("dots", "─" * (WIDTH_CH - len(title) - 3))])
        else:
            rows.append(kv_line(*item))
    return rows


def render(theme_name, stats):
    t = THEMES[theme_name]
    here = os.path.dirname(os.path.abspath(__file__))
    art = open(os.path.join(here, t["art_file"]), encoding="utf-8").read().split("\n")
    info = build_info(stats)

    body = max(len(art) * ART_LH, len(info) * LH)
    h = TOP + body + 12
    w = int(INFO_X + WIDTH_CH * CH + 24)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'viewBox="0 0 {w} {h}" font-family="ui-monospace, SFMono-Regular, Consolas, '
           f'\'Liberation Mono\', Menlo, monospace" font-size="{FONT}px">',
           f'<rect width="{w}" height="{h}" rx="12" fill="{t["bg"]}" stroke="{t["border"]}"/>']

    art_top = TOP + (body - len(art) * ART_LH) // 2
    out.append(f'<text x="{ART_X}" y="{art_top}" fill="{t["art"]}" font-size="{ART_FONT}px" '
               f'xml:space="preserve">')
    for i, line in enumerate(art):
        out.append(f'<tspan x="{ART_X}" y="{art_top + i * ART_LH}">{escape(line) or " "}</tspan>')
    out.append("</text>")

    info_top = TOP + (body - len(info) * LH) // 2
    out.append(f'<text x="{INFO_X}" y="{info_top}" fill="{t["text"]}" xml:space="preserve">')
    for i, segs in enumerate(info):
        y = info_top + i * LH
        if not segs:
            continue
        parts = "".join(f'<tspan fill="{t[k]}">{escape(v)}</tspan>' for k, v in segs)
        out.append(f'<tspan x="{INFO_X}" y="{y}">{parts}</tspan>')
    out.append("</text></svg>")
    return "\n".join(out)


if __name__ == "__main__":
    token = os.environ.get("GH_TOKEN")
    stats = fetch_stats(token) if token else None
    here = os.path.dirname(os.path.abspath(__file__))
    for name in THEMES:
        with open(os.path.join(here, f"{name}_mode.svg"), "w", encoding="utf-8") as f:
            f.write(render(name, stats))
    print("stats:", stats)

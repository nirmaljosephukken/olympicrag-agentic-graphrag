"""Build the metrics dashboard from results/metrics.json.

    python -m agr.dashboard   ->  dashboard/index.html (open in a browser) + dashboard/artifact.html (body only)
"""
from __future__ import annotations

import datetime as dt
import json

from agr.config import ROOT, SETTINGS


def build() -> None:
    data = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    data["meta"] = {"model": SETTINGS.model, "judge": SETTINGS.judge_model,
                    "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    tpl = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
    body = tpl.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    (ROOT / "dashboard" / "artifact.html").write_text(body, encoding="utf-8")
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
            + body + "\n</body>\n</html>\n")
    (ROOT / "dashboard" / "index.html").write_text(full, encoding="utf-8")
    print("wrote dashboard/index.html and dashboard/artifact.html")


if __name__ == "__main__":
    build()

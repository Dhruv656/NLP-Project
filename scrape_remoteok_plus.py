import csv
import html
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import requests

REMOTEOK_URL = "https://remoteok.com/remote-jobs.json"
WWR_URL = "https://weworkremotely.com/remote-jobs.rss"
OUTPUT_FILE = Path("remoteok_jobs.csv")


def clean_text(value):
    if not value:
        return ""
    text = html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_salary(job):
    salary_min = job.get("salary_min")
    salary_max = job.get("salary_max")
    if salary_min and salary_max and salary_min != salary_max:
        return f"${salary_min} - ${salary_max}"
    if salary_min:
        return f"${salary_min}"
    if salary_max:
        return f"${salary_max}"
    return ""


def parse_tags(job):
    tags = job.get("tags") or []
    if isinstance(tags, list):
        return ", ".join(str(tag).strip() for tag in tags if str(tag).strip())
    return str(tags).strip()


def fetch_remoteok():
    response = requests.get(REMOTEOK_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    jobs = response.json()
    if not isinstance(jobs, list):
        raise ValueError("Unexpected RemoteOK response format")
    rows = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        rows.append(
            {
                "job_title": str(job.get("position", "")).strip(),
                "company": str(job.get("company", "")).strip(),
                "location": str(job.get("location", "")).strip(),
                "tags": parse_tags(job),
                "salary": parse_salary(job),
                "date_posted": str(job.get("date", "")).strip(),
                "description": clean_text(job.get("description", "")),
                "url": str(job.get("url", "")).strip(),
                "source": "RemoteOK",
            }
        )
    return rows


def fetch_weworkremotely():
    response = requests.get(WWR_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        desc = item.findtext("description") or ""
        pub_date = item.findtext("pubDate") or ""
        rows.append(
            {
                "job_title": clean_text(title),
                "company": "",
                "location": "",
                "tags": "",
                "salary": "",
                "date_posted": clean_text(pub_date),
                "description": clean_text(desc),
                "url": clean_text(link),
                "source": "We Work Remotely",
            }
        )
    return rows


def main():
    rows = fetch_remoteok() + fetch_weworkremotely()

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "job_title",
                "company",
                "location",
                "tags",
                "salary",
                "date_posted",
                "description",
                "url",
                "source",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} jobs to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

import csv
import html
import re
from pathlib import Path

import requests

URL = "https://remoteok.com/remote-jobs.json"
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


def main():
    response = requests.get(URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    jobs = response.json()
    if not isinstance(jobs, list):
        raise ValueError("Unexpected RemoteOK response format")

    fieldnames = [
        "job_title",
        "company",
        "location",
        "tags",
        "salary",
        "date_posted",
        "description",
        "url",
    ]

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
            }
        )

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} jobs to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()

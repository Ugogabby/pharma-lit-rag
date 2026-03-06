"""
scripts/validate_pr_papers.py
──────────────────────────────
Validation script called by the GitHub Actions workflow.

What it does:
  1. Uses git diff to find new lines added to papers.jsonl in this PR
  2. Extracts PMIDs from those new lines
  3. Looks up each PMID on PubMed to verify it exists
  4. Checks the paper is relevant to your KB scope
  5. Posts a formatted comment on the PR with results
  6. Exits with code 1 (failure) if any paper fails validation
     so GitHub blocks the merge until issues are fixed

Relevance check:
  A paper passes if its title or MeSH terms contain at least one
  keyword from RELEVANCE_KEYWORDS below. This is intentionally broad —
  it catches anything in your topic space while blocking completely
  unrelated submissions.

Run locally to test:
  python scripts/validate_pr_papers.py --local data/papers.jsonl
"""

import os
import sys
import json
import subprocess
import requests
import xml.etree.ElementTree as ET
from datetime import datetime


# ── PubMed API endpoint ───────────────────────────────────────────────────────
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# ── GitHub API endpoint ───────────────────────────────────────────────────────
GITHUB_API  = "https://api.github.com"

# ── Relevance keywords ────────────────────────────────────────────────────────
# A submitted paper passes if its title contains ANY of these terms.
# Case-insensitive. Add more as your KB scope expands.
RELEVANCE_KEYWORDS = [
    # TTE / causal methods
    "target trial", "trial emulation", "sequential trial",
    "clone censor", "per protocol", "intention to treat",
    "causal inference", "estimand", "counterfactual",

    # Competing events
    "competing risk", "competing event", "subdistribution",
    "cause-specific", "fine-gray", "cumulative incidence",

    # Study design
    "active comparator", "new user design", "immortal time",
    "channeling bias", "confounding", "propensity score",
    "marginal structural", "inverse probability",
    "high dimensional propensity", "hdps",

    # Drug / exposure
    "opioid", "gabapentin", "pregabalin", "gabapentinoid",
    "benzodiazepine", "sedative", "analgesic", "naltrexone",
    "buprenorphine", "methadone",

    # Outcomes
    "overdose", "mortality", "death", "hospitalization",
    "adverse drug", "drug safety",

    # Data / methods
    "pharmacoepidemiology", "pharmacoepi",
    "claims data", "medicaid", "medicare", "administrative data",
    "real world evidence", "comparative effectiveness",
    "cohort study", "observational study",
    "health outcomes", "heor",

    # General epidemiology methods relevant to your work
    "epidemiology", "survival analysis", "hazard ratio",
    "time to event", "cox proportional",
]


def get_ncbi_api_key() -> str:
    """Returns the NCBI API key from environment, or empty string."""
    return os.environ.get("NCBI_API_KEY", "")


def find_new_pmids_from_diff(base_sha: str, head_sha: str) -> list[str]:
    """
    Uses git diff to find PMIDs added in this PR.

    Compares base branch to head branch and extracts PMIDs from
    new lines added to papers.jsonl.

    Args:
        base_sha: SHA of the base branch commit
        head_sha: SHA of the head branch (PR) commit

    Returns:
        list of PMID strings added in this PR
    """
    try:
        # Get the diff between base and head for papers.jsonl
        result = subprocess.run(
            ["git", "diff", base_sha, head_sha, "--", "data/papers.jsonl"],
            capture_output=True, text=True, check=True
        )
        diff_output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Git diff failed: {e}")
        return []

    pmids = []
    for line in diff_output.split("\n"):
        # Lines starting with + are additions (excluding +++ header lines)
        if line.startswith("+") and not line.startswith("+++"):
            content = line[1:].strip()   # remove the leading +
            if not content:
                continue
            try:
                paper = json.loads(content)
                pmid  = str(paper.get("pmid", "")).strip()
                if pmid and pmid.isdigit():
                    pmids.append(pmid)
            except json.JSONDecodeError:
                continue

    return pmids


def load_pmids_from_file(filepath: str) -> list[str]:
    """
    Loads all PMIDs from a papers.jsonl file.
    Used for local testing without git diff.

    Args:
        filepath: path to papers.jsonl

    Returns:
        list of PMID strings
    """
    pmids = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
                pmid  = str(paper.get("pmid", "")).strip()
                if pmid and pmid.isdigit():
                    pmids.append(pmid)
            except json.JSONDecodeError:
                continue
    return pmids


def validate_pmid(pmid: str, api_key: str = "") -> dict:
    """
    Validates a single PMID against PubMed.

    Checks:
      1. The PMID exists in PubMed
      2. We can retrieve its metadata
      3. The title contains at least one relevance keyword

    Args:
        pmid:    PubMed ID to validate
        api_key: NCBI API key for higher rate limits

    Returns:
        dict with keys:
          valid:    bool — True if paper passes all checks
          pmid:     the PMID
          title:    paper title (empty if not found)
          authors:  author string
          year:     publication year
          journal:  journal name
          reason:   why it failed (empty if valid)
    """
    params = {
        "db":      "pubmed",
        "id":      pmid,
        "retmode": "xml",
        "rettype": "abstract",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        response = requests.get(EFETCH_URL, params=params, timeout=20)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        article = root.find(".//PubmedArticle")

        if article is None:
            return {
                "valid":   False,
                "pmid":    pmid,
                "title":   "",
                "authors": "",
                "year":    "",
                "journal": "",
                "reason":  f"PMID {pmid} not found in PubMed",
            }

        # Extract title
        title_elem = article.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

        # Extract authors
        authors = []
        for author in article.findall(".//Author"):
            last  = author.find("LastName")
            first = author.find("ForeName")
            last_name  = last.text.strip()  if last  is not None and last.text  else ""
            first_name = first.text.strip() if first is not None and first.text else ""
            if last_name:
                authors.append(f"{last_name} {first_name}".strip())
        author_string = ", ".join(authors[:3])
        if len(authors) > 3:
            author_string += " et al."

        # Extract year
        year_elem = article.find(".//PubDate/Year")
        year = year_elem.text.strip() if year_elem is not None and year_elem.text else ""

        # Extract journal
        journal_elem = article.find(".//Journal/Title")
        journal = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else ""

        # Extract MeSH terms for relevance check
        mesh_terms = []
        for mesh in article.findall(".//MeshHeading/DescriptorName"):
            if mesh.text:
                mesh_terms.append(mesh.text.lower())

        # ── Relevance check ───────────────────────────────────────────────────
        # Check title and MeSH terms against keywords
        title_lower = title.lower()
        mesh_text   = " ".join(mesh_terms)
        search_text = title_lower + " " + mesh_text

        is_relevant = any(kw in search_text for kw in RELEVANCE_KEYWORDS)

        if not title:
            return {
                "valid":   False,
                "pmid":    pmid,
                "title":   "",
                "authors": author_string,
                "year":    year,
                "journal": journal,
                "reason":  f"Could not retrieve title for PMID {pmid}",
            }

        if not is_relevant:
            return {
                "valid":   False,
                "pmid":    pmid,
                "title":   title,
                "authors": author_string,
                "year":    year,
                "journal": journal,
                "reason":  (
                    f"Paper does not appear relevant to this KB. "
                    f"Title: '{title}'. "
                    f"Expected keywords related to pharmacoepidemiology, "
                    f"target trial emulation, opioids, competing events, "
                    f"or related methods."
                ),
            }

        return {
            "valid":   True,
            "pmid":    pmid,
            "title":   title,
            "authors": author_string,
            "year":    year,
            "journal": journal,
            "reason":  "",
        }

    except Exception as e:
        return {
            "valid":   False,
            "pmid":    pmid,
            "title":   "",
            "authors": "",
            "year":    "",
            "journal": "",
            "reason":  f"Validation error for PMID {pmid}: {str(e)}",
        }


def build_pr_comment(results: list[dict], pr_number: int) -> str:
    """
    Builds a formatted markdown comment to post on the Pull Request.

    Args:
        results:   list of validation result dicts
        pr_number: GitHub PR number

    Returns:
        markdown string for the PR comment
    """
    valid_papers   = [r for r in results if r["valid"]]
    invalid_papers = [r for r in results if not r["valid"]]
    all_passed     = len(invalid_papers) == 0

    # ── Header ────────────────────────────────────────────────────────────────
    if all_passed:
        header = "## ✅ Paper Validation Passed\n\n"
        header += f"All {len(valid_papers)} submitted paper(s) passed validation.\n\n"
    else:
        header = "## ❌ Paper Validation Failed\n\n"
        header += (
            f"{len(invalid_papers)} of {len(results)} paper(s) failed validation. "
            f"Please fix the issues below before this PR can be merged.\n\n"
        )

    # ── Valid papers table ────────────────────────────────────────────────────
    valid_section = ""
    if valid_papers:
        valid_section = "### ✓ Valid Papers\n\n"
        valid_section += "| PMID | Title | Authors | Year | Journal |\n"
        valid_section += "|------|-------|---------|------|--------|\n"
        for r in valid_papers:
            title   = r["title"][:60] + "..." if len(r["title"]) > 60 else r["title"]
            pmid_link = f"[{r['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/)"
            valid_section += f"| {pmid_link} | {title} | {r['authors']} | {r['year']} | {r['journal']} |\n"
        valid_section += "\n"

    # ── Invalid papers section ────────────────────────────────────────────────
    invalid_section = ""
    if invalid_papers:
        invalid_section = "### ✗ Failed Validation\n\n"
        for r in invalid_papers:
            pmid_link = f"[{r['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/)"
            invalid_section += f"**PMID {pmid_link}**"
            if r["title"]:
                invalid_section += f" — {r['title'][:80]}"
            invalid_section += f"\n> ❌ {r['reason']}\n\n"

    # ── Footer ────────────────────────────────────────────────────────────────
    footer = "---\n"
    footer += f"*Validated by pharma-lit-rag automated checker · {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*\n"
    footer += "*[How to contribute papers](CONTRIBUTING.md)*\n"

    return header + valid_section + invalid_section + footer


def post_pr_comment(comment: str, pr_number: int):
    """
    Posts a comment on the GitHub Pull Request.

    Uses the GITHUB_TOKEN automatically provided by GitHub Actions.
    No additional setup needed.

    Args:
        comment:   markdown string to post
        pr_number: PR number to comment on
    """
    token = os.environ.get("GITHUB_TOKEN", "")
    repo  = os.environ.get("GITHUB_REPOSITORY", "Ugogabby/pharma-lit-rag")

    if not token:
        print("No GITHUB_TOKEN found — skipping PR comment")
        print("Comment would have been:")
        print(comment)
        return

    url     = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"token {token}",
        "Accept":        "application/vnd.github.v3+json",
    }
    payload = {"body": comment}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        print(f"✓ Posted validation comment on PR #{pr_number}")
    except Exception as e:
        print(f"⚠ Could not post PR comment: {e}")


def main():
    """
    Main entry point — called by GitHub Actions or directly for testing.

    Exit codes:
      0 = all papers valid (PR can be merged)
      1 = one or more papers failed (PR blocked)
    """
    import time

    # ── Detect run mode ───────────────────────────────────────────────────────
    local_mode = "--local" in sys.argv

    if local_mode:
        # Local testing: validate all PMIDs in a given file
        filepath = sys.argv[sys.argv.index("--local") + 1]
        pmids    = load_pmids_from_file(filepath)
        print(f"Local mode: validating {len(pmids)} PMIDs from {filepath}")
    else:
        # GitHub Actions mode: find PMIDs added in this PR
        base_sha   = os.environ.get("BASE_SHA", "")
        head_sha   = os.environ.get("HEAD_SHA", "")
        pr_number  = int(os.environ.get("PR_NUMBER", "0"))

        if not base_sha or not head_sha:
            print("Missing BASE_SHA or HEAD_SHA environment variables")
            sys.exit(1)

        pmids = find_new_pmids_from_diff(base_sha, head_sha)
        print(f"Found {len(pmids)} new PMID(s) in PR #{pr_number}: {pmids}")

    if not pmids:
        print("No new PMIDs found to validate.")
        sys.exit(0)

    # ── Validate each PMID ────────────────────────────────────────────────────
    api_key = get_ncbi_api_key()
    results = []

    for pmid in pmids:
        print(f"  Validating PMID {pmid}...")
        result = validate_pmid(pmid, api_key)
        results.append(result)

        status = "✓" if result["valid"] else "✗"
        print(f"  {status} {pmid} — {result.get('title', '')[:60]}")
        if not result["valid"]:
            print(f"    Reason: {result['reason']}")

        # Respect PubMed rate limit
        time.sleep(0.4 if not api_key else 0.15)

    # ── Build and post comment ────────────────────────────────────────────────
    if not local_mode:
        pr_number = int(os.environ.get("PR_NUMBER", "0"))
        comment   = build_pr_comment(results, pr_number)
        post_pr_comment(comment, pr_number)

    # ── Exit with appropriate code ────────────────────────────────────────────
    invalid = [r for r in results if not r["valid"]]
    if invalid:
        print(f"\n❌ {len(invalid)} paper(s) failed validation.")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(results)} paper(s) passed validation.")
        sys.exit(0)


if __name__ == "__main__":
    main()
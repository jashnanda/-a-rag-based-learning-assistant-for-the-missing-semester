import subprocess
from pathlib import Path

REPO_URL = "https://github.com/missing-semester/missing-semester.git"
DATA_DIR = Path(__file__).parent.parent / "data"
COURSE_DIRS = [DATA_DIR / "_2020", DATA_DIR / "_2026"]


def clone_repo_if_needed() -> None:
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        return
    print(f"Cloning {REPO_URL} into {DATA_DIR} ...")
    subprocess.run(["git", "clone", REPO_URL, str(DATA_DIR)], check=True)


def count_code_blocks(lines) -> int:
    count = 0
    for line in lines:
        if line.startswith("```"):
            count += 1
    return count


def count_headings(lines) -> int:
    count = 0
    for line in lines:
        if line.startswith("#") and len(line) > 1 and line[1] in "# \t":
            count += 1
    return count


def analyze_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    words = len(text.split())
    return {"file": path.name, "chars": len(text), "words": words, "headings": count_headings(lines), "code_blocks": count_code_blocks(lines),}


def main() -> None:
    clone_repo_if_needed()

    files = []
    for d in COURSE_DIRS:
        files.extend(d.glob("*.md"))
    files = sorted(files)

    rows = [analyze_file(f) for f in files]

    #create our per file table
    header = f"{'File':<32}  {'Words':>8}  {'Chars':>9}  {'Headings':>8}  {'CodeBlocks':>10}"
    seperator = "-" * len(header)
    print("\n" + seperator)
    print(header)
    print(seperator)
    for r in rows:
        print(
            f"{r['file']:<32}  {r['words']:>8,}  {r['chars']:>9,}"
            f"  {r['headings']:>8}  {r['code_blocks']:>10}")
    print(seperator)
    n = len(rows)
    total_words = sum(r["words"] for r in rows)
    total_chars = sum(r["chars"] for r in rows)
    total_headings = sum(r["headings"] for r in rows)
    total_code_blocks = sum(r["code_blocks"] for r in rows)

    print(f"\nTotal .md files   : {n}")
    print(f"Total words       : {total_words:,}")
    print(f"Total characters  : {total_chars:,}")
    print(f"Total headings    : {total_headings}")
    print(f"Total code blocks : {total_code_blocks}")
    print(f"\nAverage words/file: {total_words // n:,}")
    print(f"Average chars/file: {total_chars // n:,}")
    print(f"Avg headings/file : {total_headings / n:.1f}")
    print(f"Avg code blks/file: {total_code_blocks / n:.1f}")


if __name__ == "__main__":
    main()

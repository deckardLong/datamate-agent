from pathlib import Path
from utils import clean_name

def parse_rulebook(rulebook_path: Path) -> dict:
    """
    Parse file rulebook.md thành cấu trúc JSON quy tắc tổng thể và theo từng phần cho Agent.
    """
    if not rulebook_path.exists():
        return {}

    content = rulebook_path.read_text(encoding="utf-8")
    rules = {"global_rules": [], "sections": {}}

    current_section = "general"
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            current_section = clean_name(line[2:].strip())
            rules["sections"][current_section] = []
        elif line.startswith("## "):
            current_section = clean_name(line[3:].strip())
            rules["sections"][current_section] = []
        elif line.startswith("- ") or line.startswith("* "):
            rule_text = line[2:].strip()
            if current_section in rules["sections"]:
                rules["sections"][current_section].append(rule_text)
            else:
                rules["global_rules"].append(rule_text)

    return rules

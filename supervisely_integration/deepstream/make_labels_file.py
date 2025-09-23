import json
import sys
from pathlib import Path

def main():
    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: File {json_path} not found")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    classes = [cls["title"] for cls in data.get("classes", [])]

    labels_path = json_path.parent / "labels.txt"

    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(classes))

    print(f"Labels saved to {labels_path}")

if __name__ == "__main__":
    main()

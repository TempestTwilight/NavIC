import os
import json
import glob

RAW = "/home/tempest/Documents/ISRO_Project/data/Raw_Logs"
files = sorted(glob.glob(os.path.join(RAW, "*.txt")))
out = []
for f in files:
    hdrs = {}  # record label -> header list
    version_line = None
    with open(f, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > 200:
                break
            line = line.rstrip("\n")
            if line.startswith("# Version:"):
                version_line = line[10:].strip()
            elif line.startswith("#") and "," in line:
                parts = line[2:].split(",")
                hdrs[parts[0]] = parts[1:]
    out.append({"file": os.path.basename(f), "version": version_line, "records": hdrs})

# Compare record sets and column orders across files
all_labels = sorted({lbl for r in out if isinstance(r["records"], dict) for lbl in r["records"]})
print("Total .txt files:", len(out))
print("Union of record labels:", all_labels)
# group by signature (labels + column lists)
sigs = {}
for r in out:
    sig = json.dumps(r["records"], sort_keys=False)
    sigs.setdefault(sig, []).append(r["file"])
print("Distinct header signatures:", len(sigs))
for k, v in sigs.items():
    recs = json.loads(k)
    print(
        "---signature with",
        len(v),
        "files:",
        ", ".join(sorted(set(f.split("_")[0] for f in v))),
    )
    for lab, cols in recs.items():
        print("   ", lab, "->", ",".join(cols))
    print(
        "   files:", [os.path.basename(x) for x in v][:3], "..." if len(v) > 3 else ""
    )
with open("/tmp/opencode/header_inventory.json", "w") as fh:
    json.dump(out, fh)

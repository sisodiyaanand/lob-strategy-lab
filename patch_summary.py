from pathlib import Path

OLD = '''    for name, p in rows:
        print(f"{name:14s}: {fmt(evaluate(simulate, p, args.binary, test))}")
    print(f"Log: {log_path}")
'''
NEW = '''    results = {}
    for name, p in rows:
        m = evaluate(simulate, p, args.binary, test)
        results[name] = {"params": p, "metrics": m}
        print(f"{name:14s}: {fmt(m)}")
    log({"summary": results, "heldout_seeds": len(test), "sim": sim_name, "llm": args.llm})
    print(f"Log: {log_path}")
'''

p = Path("lob_cli_lab.py")
src = p.read_text(encoding="utf-8")
if '"summary": results' in src:
    raise SystemExit("already patched")
if src.count(OLD) != 1:
    raise SystemExit("could not find the expected block; send me lob_cli_lab.py")
Path("lob_cli_lab.py.bak2").write_text(src, encoding="utf-8")
p.write_text(src.replace(OLD, NEW), encoding="utf-8")
print("patched OK (backup: lob_cli_lab.py.bak2)")
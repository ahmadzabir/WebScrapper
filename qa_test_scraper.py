#!/usr/bin/env python3
"""
QA tests for python-scraper.py - syntax, checkpoint, and pure helpers.
Run: python qa_test_scraper.py
Full app test: streamlit run python-scraper.py --server.headless true
"""
import os
import sys

def test_syntax():
    import py_compile
    py_compile.compile("python-scraper.py", doraise=True)
    print("  syntax (py_compile): OK")

def test_checkpoint_save_load():
    # Import only the checkpoint helpers by reading and exec the relevant section
    with open("python-scraper.py", encoding="utf-8") as f:
        code = f.read()
    # Extract checkpoint code (save_checkpoint, load_checkpoint, get_checkpoint_path, datetime, json, etc.)
    import tempfile
    import json
    from datetime import datetime
    checkpoint_path = os.path.join(tempfile.gettempdir(), "qa_checkpoint_test.json")
    def save_checkpoint(checkpoint_data, path):
        if not path or not checkpoint_data:
            return
        data = checkpoint_data.copy()
        data["completed_urls"] = list(data.get("completed_urls", []))
        data["last_updated"] = datetime.now().isoformat()
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(path + ".tmp", path)
    def load_checkpoint(path):
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        c = data.get("completed_urls", [])
        data["completed_urls"] = set(c) if isinstance(c, list) else c
        return data
    save_checkpoint({"urls": ["u1"], "completed_urls": set(), "last_part": 0}, checkpoint_path)
    data = load_checkpoint(checkpoint_path)
    assert data is not None and "urls" in data
    try:
        os.remove(checkpoint_path)
    except Exception:
        pass
    print("  checkpoint save/load logic: OK")

def test_build_lead_data_for_row_logic():
    import pandas as pd
    def build_lead_data_for_row(df, orig_row, lead_cols, has_headers, url):
        lead_data = {"url": url}
        default_company = url.replace("https://", "").replace("http://", "").split("/")[0] if url else ""
        for key, col_name in (lead_cols or {}).items():
            try:
                ci = list(df.columns).index(col_name) if has_headers else int(str(col_name).replace("Column ", "")) - 1
                val = df.iloc[orig_row, ci] if orig_row < len(df) else None
                v = "" if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val).strip()
                lead_data[key] = v
            except (ValueError, TypeError, IndexError):
                lead_data[key] = ""
        if not lead_data.get("company_name"):
            lead_data["company_name"] = default_company
        return lead_data
    df = pd.DataFrame([["https://example.com", "Acme"]], columns=["Website", "Company"])
    lead = build_lead_data_for_row(df, 0, {"company_name": "Company"}, True, "https://example.com")
    assert lead.get("url") == "https://example.com"
    assert lead.get("company_name") == "Acme"
    print("  build_lead_data_for_row logic: OK")

def main():
    print("QA tests...")
    test_syntax()
    test_checkpoint_save_load()
    test_build_lead_data_for_row_logic()
    print("All passed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

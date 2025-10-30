#!/usr/bin/env python3
# coding: utf-8
# python e:\AMyCode\others\competition\summer-ospp\MindSporeCodeAssistant\openwebui-pipeline\mapping_gen.py --excel "e:\path\to\报错列表.xlsx" --var-name "my_mapping" --url-backticks --sort


import argparse
import os
import re
import sys

def _ensure_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        print("Missing dependency: pandas. Install with 'pip install pandas openpyxl'", file=sys.stderr)
        sys.exit(1)

def read_table(path, sheet_name=None):
    ext = os.path.splitext(path)[1].lower()
    pd = _ensure_pandas()
    if ext in ('.xlsx', '.xls', '.xlsm'):
        df = pd.read_excel(path, sheet_name=sheet_name)
        if isinstance(df, dict):
            df = next(iter(df.values()))
    elif ext in ('.csv', '.tsv'):
        sep = '\t' if ext == '.tsv' else ','
        df = pd.read_csv(path, sep=sep)
    else:
        # Fallback: try excel then csv
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
            if isinstance(df, dict):
                df = next(iter(df.values()))
        except Exception:
            df = pd.read_csv(path)
    return df

def clean_text(x):
    if x is None:
        return None
    s = str(x).strip()
    s = s.strip("`'\" \t\r\n")
    return s

def clean_title_no_escape(x):
    # 仅保留汉字、空格、字母；不做任何转义
    if x is None:
        return None
    s = str(x).strip()
    s = re.sub(r'[^A-Za-z\u4e00-\u9fff ]+', '', s)
    return s

def extract_url(s):
    if not s:
        return s
    m = re.search(r'https?://\S+', s)
    if m:
        url = m.group(0)
        url = url.rstrip('`\'" ,)')
        return url
    return s

def build_mapping(df, title_col='标题', link_col='链接'):
    if title_col not in df.columns or link_col not in df.columns:
        raise KeyError(f"Missing expected columns: '{title_col}' and/or '{link_col}'. Found: {list(df.columns)}")

    mapping = {}
    for _, row in df.iterrows():
        # 标题仅保留汉字/空格/字母，不做转义
        title = clean_title_no_escape(row.get(title_col))
        # 链接保留为普通字符串（不使用反引号），同时提取首个 http(s) 链接
        link = clean_text(row.get(link_col))
        if not title or not link:
            continue
        link = extract_url(link)
        mapping[title] = link
    return mapping

def mapping_to_python_code(mapping, var_name='llm_docs_mapping', sort=False):
    items = mapping.items()
    if sort:
        items = sorted(items, key=lambda kv: kv[0])
    lines = [f"{var_name} = {{"]

    for k, v in items:
        # 标题不做转义；链接不使用反引号
        lines.append(f'    "{k}": "{v}",')
    lines.append("}")
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Generate a Python mapping from an Excel table.")
    parser.add_argument("--excel", required=True, help="Path to the Excel/CSV file")
    parser.add_argument("--sheet", default=None, help="Sheet name or index for Excel (optional)")
    parser.add_argument("--title-col", default="标题", help="Column name for titles")
    parser.add_argument("--link-col", default="链接", help="Column name for links")
    parser.add_argument("--var-name", default="llm_docs_mapping", help="Output variable name")
    parser.add_argument("--sort", action="store_true", help="Sort entries by title")
    parser.add_argument("--output-file", default=None, help="If set, write the mapping code to this .py file")
    args = parser.parse_args()

    df = read_table(args.excel, sheet_name=args.sheet)
    mapping = build_mapping(df, title_col=args.title_col, link_col=args.link_col)
    code = mapping_to_python_code(mapping, var_name=args.var_name, sort=args.sort)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(code)
    else:
        print(code)

if __name__ == "__main__":
    main()
    
    
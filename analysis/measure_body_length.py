#!/usr/bin/env python3
"""measure_body_length.py — how long is the body, in column-pages?

Locates the `Limitations` heading by bounding box and reports how far into the document it sits.
Everything before it counts against the venue page limit; everything from it on does not.

Counting lines out of `pdftotext` does NOT work: a table cell counts as a line, so the lines per
page move with the floats and the same body measures differently on two builds that differ only in
where a table landed.

    python3 analysis/measure_body_length.py paper/latex/acl_latex.pdf
"""
import re,subprocess,sys
xml=subprocess.run(['pdftotext','-bbox',sys.argv[1],'-'],capture_output=True,text=True).stdout
for i,pg in enumerate(re.findall(r'<page[^>]*width="([\d.]+)"[^>]*height="([\d.]+)"[^>]*>(.*?)</page>',xml,re.S),1):
    w,h,body=float(pg[0]),float(pg[1]),pg[2]
    for m in re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)"[^>]*>([^<]*)</word>',body):
        if m.group(3)=='Limitations':
            x,y=float(m.group(1)),float(m.group(2))
            col = 0.5 if x > w/2 else 0.0
            pos = (i-1) + col + 0.5*y/h
            print(f"  body ends p{i}, {'right' if col else 'left'} column, {100*y/h:.0f}% down"
                  f"  ->  {pos:.3f} column-pages")
            sys.exit(0)
print("  not located")

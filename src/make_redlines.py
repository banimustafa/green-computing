"""
Phase 10 - Redlining.

Produces the two marked-up Word deliverables from the clean document: one with the edits as
Word tracked changes, and one with the same edits highlighted for readers who prefer a flat
view. The edits are those applied between draft v1 and the shipped v1.1, both of which arose
from the consistency audit.

The approach unpacks the document, splits the run containing each changed phrase, and rebuilds
it as prefix, marked change, suffix. Splitting is necessary because a phrase visible in the
document rarely exists as its own run.
"""

import html
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
AUTHOR = "Consistency audit"
DATE = "2026-09-13T00:00:00Z"

# (text removed in v1.1, text added in v1.1). An empty removal is a pure addition.
EDITS = [
    ("the remaining 91.17 per cent of machines are unlabelled and carry 9.02 per cent of core hours.",
     "the remaining 2,457,455 machines, 91.17 per cent of the trace, are unlabelled and carry "
     "9.02 per cent of core hours, and are excluded from the workload model."),
    ("",
     ", and its design-period error of 32.01 shows that the gap is not an artefact of the "
     "held-out window"),
]


def unpack(docx, dest):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    subprocess.run(["unzip", "-q", docx, "-d", dest], check=True)
    for root, _, files in os.walk(dest):
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p):
                os.unlink(p)


def pack(src, docx):
    if os.path.exists(docx):
        os.remove(docx)
    subprocess.run(["zip", "-Xqr", os.path.abspath(docx), "."], cwd=src, check=True)


def split_run(xml, phrase, build):
    """Find the run holding `phrase`, split it, and rebuild it with `build(prefix, suffix)`."""
    esc = html.escape(phrase, quote=False)
    for m in re.finditer(r"<w:r(?: [^>]*)?>(?:(?!</w:r>).)*?</w:r>", xml, flags=re.S):
        run = m.group(0)
        tm = re.search(r"<w:t(?: [^>]*)?>(.*?)</w:t>", run, flags=re.S)
        if not tm or esc not in tm.group(1):
            continue
        text = tm.group(1)
        i = text.index(esc)
        pre, post = text[:i], text[i + len(esc):]
        rpr = re.search(r"<w:rPr>.*?</w:rPr>", run, flags=re.S)
        rpr = rpr.group(0) if rpr else ""
        new = build(pre, post, rpr)
        return xml[:m.start()] + new + xml[m.end():], True
    return xml, False


def run_xml(text, rpr="", extra=""):
    if not text:
        return ""
    return ('<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>'
            % (rpr_with(rpr, extra), html.escape(text, quote=False)))


def rpr_with(rpr, extra):
    if not extra:
        return rpr
    if rpr:
        return rpr.replace("</w:rPr>", extra + "</w:rPr>")
    return "<w:rPr>%s</w:rPr>" % extra


def make_tracked(xml, counter=[1000]):
    for removed, added in EDITS:
        def build(pre, post, rpr, removed=removed, added=added):
            counter[0] += 1
            i1, i2 = counter[0], counter[0] + 500
            out = run_xml(pre, rpr)
            if removed:
                out += ('<w:del w:id="%d" w:author="%s" w:date="%s"><w:r>%s'
                        '<w:delText xml:space="preserve">%s</w:delText></w:r></w:del>'
                        % (i2, AUTHOR, DATE, rpr, html.escape(removed, quote=False)))
            out += ('<w:ins w:id="%d" w:author="%s" w:date="%s">%s</w:ins>'
                    % (i1, AUTHOR, DATE, run_xml(added, rpr)))
            return out + run_xml(post, rpr)
        xml, ok = split_run(xml, added, build)
        print("  tracked: %s -> %s" % (added[:45], "ok" if ok else "NOT FOUND"))
    return xml


def make_highlighted(xml):
    for removed, added in EDITS:
        def build(pre, post, rpr, added=added):
            return (run_xml(pre, rpr)
                    + run_xml(added, rpr, '<w:highlight w:val="yellow"/>')
                    + run_xml(post, rpr))
        xml, ok = split_run(xml, added, build)
        print("  highlighted: %s -> %s" % (added[:45], "ok" if ok else "NOT FOUND"))
    return xml


def build(kind):
    src = os.path.join(OUT, "manuscript_clean.docx")
    work = os.path.join("/tmp", "mk_" + kind)
    unpack(src, work)
    # Coalesce adjacent identically-formatted runs, so a phrase visible in the document is
    # also a contiguous string in the XML and can be located and split.
    subprocess.run([sys.executable, "/mnt/skills/public/docx/scripts/merge_runs.py", work],
                   check=True, capture_output=True)
    p = os.path.join(work, "word", "document.xml")
    xml = open(p, encoding="utf-8").read()
    xml = make_tracked(xml) if kind == "tracked" else make_highlighted(xml)
    open(p, "w", encoding="utf-8").write(xml)
    out = os.path.join(OUT, "manuscript_%s.docx" % ("tracked_changes" if kind == "tracked"
                                                    else "changes_highlighted"))
    pack(work, out)
    print("wrote", out)


if __name__ == "__main__":
    for k in ("tracked", "highlighted"):
        print(k)
        build(k)

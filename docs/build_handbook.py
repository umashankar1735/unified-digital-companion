#!/usr/bin/env python3
"""Build the UDC interview handbook DOCX from its Markdown source."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "UDC_A_TO_Z_INTERVIEW_HANDBOOK.md"
OUTPUT = ROOT / "UDC_A_TO_Z_INTERVIEW_HANDBOOK.docx"

BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "1F2933"
MUTED = "5B6770"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F4F6F8"
WHITE = "FFFFFF"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)


def set_table_fixed_layout(table) -> None:
    tbl_pr = table._tbl.tblPr
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    indent = tbl_pr.find(qn("w:tblInd"))
    if indent is None:
        indent = OxmlElement("w:tblInd")
        tbl_pr.append(indent)
    indent.set(qn("w:w"), "120")
    indent.set(qn("w:type"), "dxa")


def set_cell_width(cell, twips: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(twips))
    tc_w.set(qn("w:type"), "dxa")


def add_field(paragraph, instruction: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])


def keep_together(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_pr.append(OxmlElement("w:keepNext"))
    p_pr.append(OxmlElement("w:keepLines"))


def add_inline(paragraph, text: str, *, base_bold=False, base_italic=False) -> None:
    """Add a conservative subset of Markdown inline formatting."""
    token_re = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))")
    position = 0
    for match in token_re.finditer(text):
        if match.start() > position:
            run = paragraph.add_run(text[position:match.start()])
            run.bold = base_bold
            run.italic = base_italic
        token = match.group(0)
        if token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            run.font.name = "SFMono-Regular"
            run.font.size = Pt(8.5)
            run.font.color.rgb = RGBColor.from_string(DARK_BLUE)
        elif token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
            run.italic = base_italic
        elif token.startswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
            run.bold = base_bold
        else:
            link_match = re.match(r"\[([^\]]+)\]\(([^)]+)\)", token)
            label, url = link_match.groups()
            run = paragraph.add_run(f"{label} ({url})")
            run.font.color.rgb = RGBColor.from_string(BLUE)
            run.underline = True
        position = match.end()
    if position < len(text):
        run = paragraph.add_run(text[position:])
        run.bold = base_bold
        run.italic = base_italic


def configure_styles(document: Document) -> None:
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.16
    normal.paragraph_format.widow_control = True

    heading_specs = {
        "Title": (29, INK, 0, 12),
        "Subtitle": (15, BLUE, 0, 12),
        "Heading 1": (16, BLUE, 18, 8),
        "Heading 2": (13, BLUE, 13, 6),
        "Heading 3": (11.5, DARK_BLUE, 10, 4),
    }
    for name, (size, color, before, after) in heading_specs.items():
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.widow_control = True

    for name in ("List Bullet", "List Number"):
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(10.5)
        style.paragraph_format.left_indent = Inches(0.375)
        style.paragraph_format.first_line_indent = Inches(-0.188)
        style.paragraph_format.space_after = Pt(3)
        style.paragraph_format.line_spacing = 1.12

    custom_specs = {
        "UDC Metadata": (9.5, MUTED, False),
        "UDC Quote": (10, DARK_BLUE, False),
        "UDC Code": (8.25, INK, False),
        "UDC Caption": (8.5, MUTED, True),
    }
    for name, (size, color, italic) in custom_specs.items():
        if name not in styles:
            style = styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        else:
            style = styles[name]
        style.font.name = "Calibri" if name != "UDC Code" else "SFMono-Regular"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.italic = italic
        style.paragraph_format.space_after = Pt(5)
        style.paragraph_format.widow_control = True
    quote = styles["UDC Quote"]
    quote.paragraph_format.left_indent = Inches(0.28)
    quote.paragraph_format.right_indent = Inches(0.12)
    quote.paragraph_format.space_before = Pt(5)
    quote.paragraph_format.space_after = Pt(7)
    code = styles["UDC Code"]
    code.paragraph_format.left_indent = Inches(0.18)
    code.paragraph_format.right_indent = Inches(0.12)
    code.paragraph_format.space_before = Pt(3)
    code.paragraph_format.space_after = Pt(6)
    code.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE


def add_cover(document: Document) -> None:
    for _ in range(3):
        document.add_paragraph()
    p = document.add_paragraph(style="Title")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("Unified Digital Companion (UDC)")
    p = document.add_paragraph(style="Subtitle")
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("A-to-Z Project Handbook and Interview Guide")
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    run = p.add_run("LEGAL AWARENESS  |  UNIVERSITY SUPPORT  |  RETRIEVAL-AUGMENTED QA")
    run.font.size = Pt(9)
    run.font.bold = True
    run.font.color.rgb = RGBColor.from_string(BLUE)

    for _ in range(3):
        document.add_paragraph()
    meta = [
        ("Repository", "Final Year - PROJECT"),
        ("Inspection date", "21 August 2026"),
        ("Prepared for", "Interview study, demonstration, and engineering handoff"),
        ("Evidence", "Source, configuration, datasets, model/index metadata, runtime checks, papers, and slides"),
    ]
    table = document.add_table(rows=len(meta), cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_fixed_layout(table)
    # Mark the first row for assistive technology; this is a compact metadata table.
    set_repeat_table_header(table.rows[0])
    for row, (key, value) in zip(table.rows, meta):
        set_cell_width(row.cells[0], 2100)
        set_cell_width(row.cells[1], 6500)
        set_cell_shading(row.cells[0], LIGHT_BLUE)
        for cell in row.cells:
            set_cell_margins(cell, top=100, bottom=100)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        row.cells[0].paragraphs[0].add_run(key).bold = True
        row.cells[1].paragraphs[0].add_run(value)

    p = document.add_paragraph()
    p.paragraph_format.space_before = Pt(20)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Evidence-first documentation: implemented behavior is distinguished from research-target claims.")
    r.italic = True
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor.from_string(MUTED)
    document.add_page_break()


def add_header_footer(section) -> None:
    header = section.header
    p = header.paragraphs[0]
    p.text = "UDC  |  A-to-Z Project Handbook"
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in p.runs:
        run.font.name = "Calibri"
        run.font.size = Pt(8)
        run.font.bold = True
        run.font.color.rgb = RGBColor.from_string(MUTED)
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Page ")
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(MUTED)
    add_field(p, "PAGE")


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        parts = [part.strip() for part in lines[i].strip().strip("|").split("|")]
        rows.append(parts)
        i += 1
    if len(rows) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
        rows.pop(1)
    return rows, i


def add_table(document: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    column_count = max(len(row) for row in rows)
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    table = document.add_table(rows=len(normalized), cols=column_count)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_fixed_layout(table)

    weights = []
    for column in range(column_count):
        max_len = max(len(row[column]) for row in normalized)
        weights.append(max(10, min(max_len, 45)))
    total = sum(weights)
    widths = [max(1000, round(9360 * weight / total)) for weight in weights]
    delta = 9360 - sum(widths)
    widths[-1] += delta

    for row_index, row_data in enumerate(normalized):
        row = table.rows[row_index]
        if row_index == 0:
            set_repeat_table_header(row)
        for column, text in enumerate(row_data):
            cell = row.cells[column]
            set_cell_width(cell, widths[column])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
            if row_index == 0:
                set_cell_shading(cell, LIGHT_BLUE)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            add_inline(p, text, base_bold=(row_index == 0))


def add_markdown(document: Document, markdown: str) -> None:
    lines = markdown.splitlines()
    # Cover supplies the first title and metadata. Begin at the Contents heading.
    try:
        i = lines.index("# Contents")
    except ValueError:
        i = 0
    in_code = False
    code_lines: list[str] = []
    first_h1 = True
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if stripped.startswith("```"):
            if in_code:
                p = document.add_paragraph(style="UDC Code")
                p.paragraph_format.keep_together = True
                for line_index, code_line in enumerate(code_lines):
                    if line_index:
                        p.add_run().add_break()
                    p.add_run(code_line)
                p_pr = p._p.get_or_add_pPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:fill"), LIGHT_GRAY)
                p_pr.append(shd)
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_lines.append(raw)
            i += 1
            continue
        if not stripped or stripped == "---":
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            rows, i = parse_table(lines, i)
            add_table(document, rows)
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2)
            if level == 1:
                if not first_h1:
                    document.add_page_break()
                first_h1 = False
            p = document.add_paragraph(style=f"Heading {level}")
            add_inline(p, text)
            keep_together(p)
            i += 1
            continue

        if stripped.startswith(">"):
            p = document.add_paragraph(style="UDC Quote")
            add_inline(p, stripped.lstrip("> "))
            p_pr = p._p.get_or_add_pPr()
            border = OxmlElement("w:pBdr")
            left = OxmlElement("w:left")
            left.set(qn("w:val"), "single")
            left.set(qn("w:sz"), "18")
            left.set(qn("w:space"), "8")
            left.set(qn("w:color"), BLUE)
            border.append(left)
            p_pr.append(border)
            i += 1
            continue

        bullet = re.match(r"^\s*[-*]\s+(.+)$", raw)
        numbered = re.match(r"^\s*\d+\.\s+(.+)$", raw)
        if bullet:
            p = document.add_paragraph(style="List Bullet")
            add_inline(p, bullet.group(1))
            i += 1
            continue
        if numbered:
            p = document.add_paragraph(style="List Number")
            add_inline(p, numbered.group(1))
            i += 1
            continue

        p = document.add_paragraph()
        add_inline(p, stripped)
        i += 1


def set_document_properties(document: Document) -> None:
    props = document.core_properties
    props.title = "Unified Digital Companion (UDC): A-to-Z Project Handbook and Interview Guide"
    props.subject = "Source-grounded documentation for the legal-awareness and university-support prototype"
    props.author = "Project documentation generated from repository inspection"
    props.keywords = "UDC, legal awareness, university support, Flask, FastAPI, FAISS, BM25, T5, InLegalBERT, interview"
    props.comments = "Implementation behavior is distinguished from claims in legacy research artifacts."


def build() -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    document = Document()
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.82)
    section.bottom_margin = Inches(0.78)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)
    configure_styles(document)
    set_document_properties(document)
    add_header_footer(section)
    add_cover(document)
    add_markdown(document, markdown)
    document.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()

# -*- coding: utf-8 -*-
"""
将 实验三报告.md 转换为规范排版的 实验三报告.docx:
- 标题居中、各级标题分级、正文首行缩进、行距1.1
- Markdown 表格 -> 带边框的真表格 (表头加粗底纹, 单元格8pt居中)
- **加粗** / `代码` 行内解析; URL 自动转超链接
- 自动插入图表并加图注
运行: python make_report_docx.py
"""

import os
import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

MD_FILE = '实验三报告.md'
DOCX_FILE = os.environ.get('DOCX_OUT', '实验三报告.docx')

# 图表插入计划: 与 md 中 ![](...) 出现顺序一一对应 (图片路径, 宽度cm, 图注)
FIGURES = [
    ('figures/curves.png', 13.5,
     '图1 三组实验的训练/验证损失与准确率曲线'),
    ('figures/confusion_cnn_improved.png', 9.5,
     '图2 增强组 CNN 在测试集上的混淆矩阵'),
    ('figures/misclassified_cnn_improved.png', 11.5,
     '图3 增强组 CNN 的典型误分类样本（真：真实标签；预：模型预测）'),
    ('figures/gradcam_cnn_improved.png', 13.5,
     '图4 增强组 CNN 的 Grad-CAM 可视化（上：原图；下：类别热力图）'),
]

BODY_FONT = '宋体'
HEAD_FONT = '黑体'
CODE_FONT = 'Consolas'
WEST_FONT = 'Times New Roman'


def set_run_font(run, eastasia, size, bold=False, color=None, west=WEST_FONT):
    run.font.name = west
    run._element.rPr.rFonts.set(qn('w:eastAsia'), eastasia)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = color


def add_hyperlink(paragraph, url, size):
    """在段落中插入可点击的超链接."""
    from docx.opc.constants import RELATIONSHIP_TYPE
    r_id = paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK,
                                    is_external=True)
    link = OxmlElement('w:hyperlink')
    link.set(qn('r:id'), r_id)
    run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '0563C1')
    rPr.append(color)
    underline = OxmlElement('w:u')
    underline.set(qn('w:val'), 'single')
    rPr.append(underline)
    sz = OxmlElement('w:sz')
    sz.set(qn('w:val'), str(int(size * 2)))
    rPr.append(sz)
    run.append(rPr)
    t = OxmlElement('w:t')
    t.text = url
    run.append(t)
    link.append(run)
    paragraph._p.append(link)


def add_runs(paragraph, text, eastasia, size, base_bold=False):
    """解析行内 **加粗**、`代码` 与 URL 超链接, 写入段落."""
    for seg in re.split(r'(\*\*.*?\*\*|`.*?`|https?://[^\s）)】]+)', text):
        if not seg:
            continue
        if seg.startswith('**') and seg.endswith('**'):
            run = paragraph.add_run(seg[2:-2])
            set_run_font(run, eastasia, size, bold=True)
        elif seg.startswith('`') and seg.endswith('`'):
            run = paragraph.add_run(seg[1:-1])
            set_run_font(run, CODE_FONT, size - 0.5, bold=base_bold,
                         west=CODE_FONT)
        elif seg.startswith('http'):
            add_hyperlink(paragraph, seg, size)
        else:
            run = paragraph.add_run(seg)
            set_run_font(run, eastasia, size, bold=base_bold)


def add_table(doc, rows):
    """把 markdown 表格行列表转成 Word 表格."""
    data = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
    header, body = data[0], data[2:]          # 跳过 |---|---| 分隔行
    table = doc.add_table(rows=len(body) + 1, cols=len(header))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(header):
        cell = table.cell(0, j)
        cell.text = ''
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_runs(p, h, HEAD_FONT, 8, base_bold=True)
        shd = cell._tc.get_or_add_tcPr().makeelement(qn('w:shd'), {
            qn('w:val'): 'clear', qn('w:fill'): 'D9E1F2'})
        cell._tc.get_or_add_tcPr().append(shd)
    # 表头跨页重复，所有行禁止跨页拆分
    for r_idx, row in enumerate(table.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        if r_idx == 0:
            trPr.append(OxmlElement('w:tblHeader'))
    for r_idx, row in enumerate(table.rows):
        if r_idx < len(table.rows) - 1:
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.keep_with_next = True

    for i, row in enumerate(body, 1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ''
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, val, BODY_FONT, 8)
    return table


def main():
    lines = open(MD_FILE, encoding='utf-8').read().splitlines()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)   # A4
    sec.top_margin = sec.bottom_margin = Cm(2.2)
    sec.left_margin = sec.right_margin = Cm(2.6)

    fig_idx, table_buf = 0, []
    last_body = None

    def flush_table():
        nonlocal table_buf
        if table_buf:
            add_table(doc, table_buf)
            spacer = doc.add_paragraph()
            set_run_font(spacer.add_run(''), BODY_FONT, 2)
            table_buf = []

    def add_figure(path, width, caption):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_after = Pt(2)
        p.add_run().add_picture(path, width=Cm(width))
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_after = Pt(4)
        set_run_font(c.add_run(caption), BODY_FONT, 8.5,
                     color=RGBColor(0x59, 0x59, 0x59))

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush_table()
            continue
        if line.startswith('|'):
            if last_body is not None:
                last_body.paragraph_format.keep_with_next = True
            table_buf.append(line)
            continue
        flush_table()

        if line.startswith('# '):                      # 标题
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            add_runs(p, line[2:], HEAD_FONT, 15, base_bold=True)
        elif line.startswith('## '):                   # 一级节
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            add_runs(p, line[3:], HEAD_FONT, 12, base_bold=True)
        elif line.startswith('### '):                  # 二级节
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            add_runs(p, line[4:], HEAD_FONT, 10.5, base_bold=True)
        elif line.startswith('!['):                    # 图片
            m = re.match(r'!\[.*?\]\((.+?)\)', line)
            fig_idx += 1
            add_figure(m.group(1), FIGURES[fig_idx - 1][1],
                       FIGURES[fig_idx - 1][2])
        else:                                          # 正文
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.1
            p.paragraph_format.space_after = Pt(2)
            if not re.match(r'^\d+\.', line):          # 编号列表项不缩进
                p.paragraph_format.first_line_indent = Pt(20)
            add_runs(p, line, BODY_FONT, 9.5)
            last_body = p

    flush_table()

    doc.core_properties.title = '实验三实验报告：CIFAR-10 图像分类'
    doc.core_properties.author = '杨乐文'
    doc.core_properties.subject = '当代人工智能实验 实验三 图像分类'
    doc.save(DOCX_FILE)
    print(f'已生成 {DOCX_FILE}: 段落{len(doc.paragraphs)}, '
          f'表格{len(doc.tables)}, 图片{len(doc.inline_shapes)}')


if __name__ == '__main__':
    main()

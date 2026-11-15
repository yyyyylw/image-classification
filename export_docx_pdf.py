"""用本机 Word 将实验三报告 DOCX 导出为 PDF（沿用实验一/二的排版流程）。"""

import os
import sys

import win32com.client


def main() -> None:
    src = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "实验三报告.docx")
    dst = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "实验三报告.pdf")

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(src, ReadOnly=False)
        doc.SaveAs2(dst, FileFormat=17)  # wdFormatPDF
        doc.Close(False)
    finally:
        word.Quit()

    print(dst)


if __name__ == "__main__":
    main()

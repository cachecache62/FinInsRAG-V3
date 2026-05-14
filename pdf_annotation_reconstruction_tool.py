"""
PDF 文档解析信息标注和还原工具

此工具提供以下功能：
1. 将提取的文本信息标注在原 PDF 文档上
2. 根据提取的信息生成标注后的新 PDF 文档
3. 根据解析结果重建文档结构（文本 + 表格）

注意：此为独立实现，不依赖项目内部代码。
"""

import os
import json
from typing import List, Dict, Any, Tuple
from io import BytesIO
import fitz  # PyMuPDF for PDF manipulation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import red, blue, green
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
import pdfkit  # For HTML to PDF conversion
from PIL import Image
import numpy as np


class PDFAnnotationTool:
    """PDF 文档标注工具"""

    def __init__(self):
        self.doc = None

    def load_pdf(self, pdf_path: str) -> None:
        """加载 PDF 文档"""
        self.doc = fitz.open(pdf_path)

    def load_pdf_from_bytes(self, pdf_bytes: BytesIO) -> None:
        """从字节流加载 PDF 文档"""
        self.doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    def annotate_text_blocks(self, text_blocks: List[Dict[str, Any]],
                           output_path: str) -> None:
        """
        在 PDF 上标注文本块

        Args:
            text_blocks: 文本块列表，每个块包含坐标和文本信息
            output_path: 输出 PDF 路径
        """
        if not self.doc:
            raise ValueError("请先加载 PDF 文档")

        # 为每个页面创建标注
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            # 过滤当前页面的文本块
            page_blocks = [block for block in text_blocks
                          if block.get('page_number', 0) == page_num]

            for block in page_blocks:
                # 获取文本块坐标
                x0, y0, x1, y1 = block.get('bbox', [0, 0, 0, 0])

                # 创建矩形标注
                rect = fitz.Rect(x0, y0, x1, y1)

                # 添加高亮标注
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=green, fill=green)
                highlight.set_opacity(0.3)

                # 添加文本注释
                text = block.get('text', '')[:50] + '...' if len(block.get('text', '')) > 50 else block.get('text', '')
                comment = page.add_text_annot((x0, y0 - 20), text)
                comment.set_colors(stroke=blue)

        # 保存标注后的 PDF
        self.doc.save(output_path)
        print(f"标注后的 PDF 已保存到: {output_path}")

    def annotate_tables(self, tables: List[Dict[str, Any]],
                       output_path: str) -> None:
        """
        在 PDF 上标注表格

        Args:
            tables: 表格列表
            output_path: 输出 PDF 路径
        """
        if not self.doc:
            raise ValueError("请先加载 PDF 文档")

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]

            # 过滤当前页面的表格
            page_tables = [table for table in tables
                          if table.get('page_number', 0) == page_num]

            for table in page_tables:
                # 获取表格坐标
                x0, y0, x1, y1 = table.get('bbox', [0, 0, 0, 0])

                # 创建矩形标注
                rect = fitz.Rect(x0, y0, x1, y1)

                # 添加表格边框标注
                border = page.add_rect_annot(rect)
                border.set_colors(stroke=red)
                border.set_linewidth(2)

                # 添加表格注释
                comment = page.add_text_annot((x0, y0 - 30), "表格区域")
                comment.set_colors(stroke=red)

        # 保存标注后的 PDF
        self.doc.save(output_path)
        print(f"表格标注后的 PDF 已保存到: {output_path}")

    def close(self):
        """关闭文档"""
        if self.doc:
            self.doc.close()


class DocumentReconstructionTool:
    """文档重建工具"""

    def __init__(self):
        self.styles = getSampleStyleSheet()

    def reconstruct_from_parsed_data(self, parsed_data: Dict[str, Any],
                                   output_path: str) -> None:
        """
        根据解析数据重建文档

        Args:
            parsed_data: 解析后的数据，包含文本块和表格
            output_path: 输出文件路径
        """
        # 创建 PDF 文档
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []

        # 添加标题
        title = parsed_data.get('title', '重建文档')
        story.append(Paragraph(f"<b>{title}</b>", self.styles['Title']))

        # 处理文本块
        text_blocks = parsed_data.get('text_blocks', [])
        for block in text_blocks:
            text = block.get('text', '')
            if text.strip():
                # 根据布局类型添加样式
                layout_type = block.get('layout_type', 'text')
                if layout_type == 'title':
                    story.append(Paragraph(text, self.styles['Heading1']))
                elif layout_type == 'header':
                    story.append(Paragraph(text, self.styles['Heading2']))
                else:
                    story.append(Paragraph(text, self.styles['Normal']))

        # 处理表格
        tables = parsed_data.get('tables', [])
        for table_data in tables:
            if 'html' in table_data:
                # 如果是 HTML 表格，转换为文本表格
                html_table = table_data['html']
                # 简单转换 HTML 表格为文本（实际实现可能需要更复杂的解析）
                story.append(Paragraph("表格内容:", self.styles['Heading3']))
                story.append(Paragraph(html_table, self.styles['Normal']))
            elif 'data' in table_data:
                # 如果是结构化表格数据
                table_content = table_data['data']
                if isinstance(table_content, list) and table_content:
                    # 创建 ReportLab 表格
                    table = Table(table_content)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), '#d0d0d0'),
                        ('TEXTCOLOR', (0, 0), (-1, 0), '#000000'),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 14),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), '#f0f0f0'),
                        ('GRID', (0, 0), (-1, -1), 1, '#000000'),
                    ]))
                    story.append(table)

        # 生成 PDF
        doc.build(story)
        print(f"重建文档已保存到: {output_path}")

    def create_html_from_parsed_data(self, parsed_data: Dict[str, Any],
                                   output_path: str) -> None:
        """
        根据解析数据生成 HTML 文档

        Args:
            parsed_data: 解析后的数据
            output_path: 输出 HTML 文件路径
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{parsed_data.get('title', '重建文档')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .title {{ font-size: 24px; font-weight: bold; margin-bottom: 20px; }}
                .text-block {{ margin-bottom: 15px; }}
                .table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="title">{parsed_data.get('title', '重建文档')}</div>
        """

        # 添加文本块
        text_blocks = parsed_data.get('text_blocks', [])
        for block in text_blocks:
            text = block.get('text', '')
            layout_type = block.get('layout_type', 'text')
            css_class = 'text-block'
            if layout_type == 'title':
                css_class += ' title'
            html_content += f'<div class="{css_class}">{text}</div>\n'

        # 添加表格
        tables = parsed_data.get('tables', [])
        for table_data in tables:
            if 'html' in table_data:
                html_content += table_data['html'] + '\n'
            elif 'data' in table_data:
                table_content = table_data['data']
                if isinstance(table_content, list) and table_content:
                    html_content += '<table class="table">\n'
                    for i, row in enumerate(table_content):
                        tag = 'th' if i == 0 else 'td'
                        html_content += '<tr>\n'
                        for cell in row:
                            html_content += f'<{tag}>{cell}</{tag}>\n'
                        html_content += '</tr>\n'
                    html_content += '</table>\n'

        html_content += """
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML 文档已保存到: {output_path}")

    def convert_html_to_pdf(self, html_path: str, pdf_path: str) -> None:
        """
        将 HTML 转换为 PDF

        Args:
            html_path: HTML 文件路径
            pdf_path: 输出 PDF 路径
        """
        try:
            pdfkit.from_file(html_path, pdf_path)
            print(f"HTML 已转换为 PDF: {pdf_path}")
        except Exception as e:
            print(f"HTML 转 PDF 失败: {e}")


def demo_usage():
    """演示使用方法"""

    # 示例解析数据（模拟项目中的解析结果）
    sample_parsed_data = {
        'title': '示例文档',
        'text_blocks': [
            {'text': '这是一个标题', 'layout_type': 'title', 'bbox': [100, 100, 300, 120]},
            {'text': '这是正文内容第一段。', 'layout_type': 'text', 'bbox': [100, 140, 500, 160]},
            {'text': '这是正文内容第二段。', 'layout_type': 'text', 'bbox': [100, 180, 500, 200]},
        ],
        'tables': [
            {
                'data': [
                    ['姓名', '年龄', '城市'],
                    ['张三', '25', '北京'],
                    ['李四', '30', '上海']
                ],
                'bbox': [100, 220, 400, 280]
            }
        ]
    }

    # 1. 文档重建
    reconstructor = DocumentReconstructionTool()

    # 生成 PDF
    reconstructor.reconstruct_from_parsed_data(
        sample_parsed_data,
        'reconstructed_document.pdf'
    )

    # 生成 HTML
    reconstructor.create_html_from_parsed_data(
        sample_parsed_data,
        'reconstructed_document.html'
    )

    # HTML 转 PDF
    reconstructor.convert_html_to_pdf(
        'reconstructed_document.html',
        'reconstructed_from_html.pdf'
    )

    # 2. PDF 标注（需要原始 PDF 文件）
    # annotator = PDFAnnotationTool()
    # annotator.load_pdf('original_document.pdf')
    # annotator.annotate_text_blocks(sample_parsed_data['text_blocks'], 'annotated_document.pdf')
    # annotator.annotate_tables(sample_parsed_data['tables'], 'annotated_tables.pdf')
    # annotator.close()

    print("演示完成！")


if __name__ == "__main__":
    demo_usage()

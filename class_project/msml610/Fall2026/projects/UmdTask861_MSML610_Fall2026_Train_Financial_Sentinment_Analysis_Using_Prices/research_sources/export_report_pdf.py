"""Export the existing English research report using local ReportLab only."""
import argparse
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, Preformatted, Image, PageBreak


def inline(text):
    text=escape(text)
    text=re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',r'<link href="\2" color="#235c91">\1</link>',text)
    text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
    return re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)


def export(source,output,chart_dir=None,chart_title='Appendix: inspected A development slice'):
    source,output=Path(source),Path(output)
    if output.exists() or output.is_symlink():raise FileExistsError('Choose a new PDF output')
    output.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ResearchBody',parent=styles['BodyText'],fontSize=9,leading=12,spaceAfter=6))
    styles.add(ParagraphStyle(name='ResearchCell',parent=styles['BodyText'],fontSize=7,leading=9,spaceAfter=0))
    styles.add(ParagraphStyle(name='ResearchCode',fontName='Courier',fontSize=6.4,leading=8,spaceAfter=8))
    body=[];lines=source.read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            body.append(Preformatted('\n'.join(code),styles['ResearchCode'],maxLineLength=105));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                cells=[s.strip() for s in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',s or ' ') for s in cells):rows.append(cells)
                i+=1
            count=len(rows[0]);width=499
            widths=([140]+[(width-140)/(count-1)]*(count-1)) if count>2 else [175,width-175]
            table=LongTable([[Paragraph(inline(v),styles['ResearchCell']) for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce8f2')),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#c5cdd4')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f9')]),
                ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            body.extend([table,Spacer(1,9)]);continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));style='Title' if level==1 else ('Heading1' if level==2 else 'Heading2')
            body.append(Paragraph(inline(line.lstrip('#').strip()),styles[style]));i+=1;continue
        paragraph=[line];i+=1
        if not line.startswith('- '):
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','- ')):
                paragraph.append(lines[i]);i+=1
        body.append(Paragraph(inline(' '.join(paragraph)),styles['ResearchBody']))
    if chart_dir:
        body.extend([PageBreak(),Paragraph(inline(chart_title),styles['Heading1'])])
        for name in ['mae_comparison.png','cost_scenarios.png']:
            p=Path(chart_dir)/name
            image=Image(str(p));image.drawHeight=image.imageHeight/image.imageWidth*499;image.drawWidth=499
            body.extend([image,Spacer(1,12)])
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#52616e'))
        canvas.drawString(48,25,'MSML610 | Local research evidence | October 21, 2026')
        canvas.drawRightString(A4[0]-48,25,f'Page {doc.page}')
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=42,
        title='MSML610 Financial Sentiment Using Prices — Research Report',author='Group 19')
    doc.build(body,onFirstPage=footer,onLaterPages=footer)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--chart-dir');p.add_argument('--chart-title',default='Appendix: inspected A development slice')
    a=p.parse_args();print(export(a.source,a.output,a.chart_dir,a.chart_title))

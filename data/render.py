"""Draw an invoice as a PDF in one of four vendor layouts, and make scanned-looking copies.

fpdf2 with its built-in fonts keeps this pure-pip (no browser, no system libraries), at the
price of Latin-1 text only. Each layout uses its own labels, column order, date style and
number style, so the model can't rely on one fixed template.
"""

import io
import random
from datetime import UTC, date, datetime
from decimal import Decimal

import pypdfium2 as pdfium
from fpdf import FPDF, FontFace
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

FIXED_DATE = datetime(2026, 1, 1, tzinfo=UTC)  # PDF metadata date, so output is reproducible
SCAN_DPI = 130
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
BUYER = ["Bluegate Retail Group", "Accounts Payable", "400 Meridian Way", "Columbus, OH 43215"]


class InvoicePDF(FPDF):
    """FPDF with a 'Page X of Y' footer, which tells the model a document continues."""

    footer_font = "Helvetica"

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(self.footer_font, size=8)
        self.cell(0, 6, f"Page {self.page_no()} of {{nb}}", align="C")


def fmt_money(value: Decimal, style: str) -> str:
    """1234.5 -> '1,234.50' (us), '$1,234.50' (us_symbol), '1.234,50' (eu), '1234.50' (plain)."""
    us = f"{value:,.2f}"
    if style == "us_symbol":
        return "$" + us
    if style == "eu":
        return us.replace(",", " ").replace(".", ",").replace(" ", ".")
    if style == "plain":
        return f"{value:.2f}"
    return us


def fmt_date(d: date, style: str) -> str:
    """Spelled out by hand: strftime month names depend on the machine's locale."""
    if style == "us":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "eu":
        return f"{d.day:02d}.{d.month:02d}.{d.year}"
    if style == "long":
        return f"{d.day} {MONTHS[d.month - 1]} {d.year}"
    return d.isoformat()


def fmt_rate(rate: str) -> str:
    """'0.0825' -> '8.25'."""
    return f"{(Decimal(rate) * 100).normalize():f}"


def invoice_pdf(inv: dict, vendor: dict, note: str) -> tuple[bytes, int]:
    """Render one invoice. Returns (pdf bytes, page count)."""
    pdf = InvoicePDF(format="Letter")
    pdf.set_creation_date(FIXED_DATE)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    LAYOUTS[vendor["layout"]](pdf, inv, vendor)
    totals_and_note(pdf, inv, vendor, note)
    return bytes(pdf.output()), pdf.pages_count


def money_row(vendor: dict, line: dict) -> list[str]:
    """Qty, unit price and line total as this vendor prints them."""
    style = vendor["money"]
    return [f"{line['qty']:f}", fmt_money(line["unit_price"], style),
            fmt_money(line["line_total"], style)]


def classic(pdf: FPDF, inv: dict, vendor: dict) -> None:
    """Vendor top-left, meta block top-right, SKU first."""
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(120, 9, vendor["name"])
    pdf.set_font(size=22)
    pdf.cell(0, 9, "INVOICE", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    meta = [("Invoice No.", inv["invoice_number"]),
            ("Invoice Date", fmt_date(inv["invoice_date"], vendor["dates"])),
            ("PO Number", inv["po_number"])]
    for address_line, (label, value) in zip(vendor["address"] + [""], meta, strict=True):
        pdf.cell(120, 5, address_line)
        pdf.cell(30, 5, label + ":")
        pdf.cell(0, 5, value, align="R", new_x="LMARGIN", new_y="NEXT")
    bill_to(pdf, "Bill to")
    with pdf.table(col_widths=(22, 80, 14, 25, 25),
                   text_align=("LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT"),
                   headings_style=FontFace(emphasis="B", fill_color=(225, 225, 225))) as table:
        table.row(["SKU", "Description", "Qty", "Unit Price", "Amount"])
        for line in inv["lines"]:
            table.row([line["sku"], line["description"], *money_row(vendor, line)])


def band(pdf: FPDF, inv: dict, vendor: dict) -> None:
    """Dark header band, meta in three boxes, description before the item code."""
    pdf.set_fill_color(28, 58, 92)
    pdf.rect(0, 0, pdf.w, 28, style="F")
    pdf.set_fill_color(255, 255, 255)  # or the table body inherits the band colour
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(15, 8)
    pdf.cell(120, 8, vendor["name"])
    pdf.set_font(size=12)
    pdf.cell(0, 8, "TAX INVOICE", align="R")
    pdf.set_font("Helvetica", size=8)
    pdf.set_xy(15, 17)
    pdf.cell(0, 5, " | ".join(vendor["address"]))
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(15, 34)
    pdf.set_font("Helvetica", size=9)
    boxes = [f"Invoice: {inv['invoice_number']}",
             f"Dated: {fmt_date(inv['invoice_date'], vendor['dates'])}",
             f"Order ref: {inv['po_number']}"]
    for text in boxes:
        pdf.cell(62, 8, text, border=1)
    pdf.ln(10)
    bill_to(pdf, "Customer")
    with pdf.table(col_widths=(80, 26, 18, 28, 30),
                   text_align=("LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT"),
                   headings_style=FontFace(emphasis="B", color=(255, 255, 255),
                                           fill_color=(28, 58, 92))) as table:
        table.row(["Description", "Item code", "Quantity", "Price", "Line total"])
        for line in inv["lines"]:
            table.row([line["description"], line["sku"], *money_row(vendor, line)])


def european(pdf: FPDF, inv: dict, vendor: dict) -> None:
    """Bilingual French/English labels, comma decimals, HT/TVA/TTC totals."""
    pdf.footer_font = "Times"
    pdf.set_font("Times", "B", 15)
    pdf.cell(110, 8, vendor["name"])
    pdf.cell(0, 8, "FACTURE / INVOICE", align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", size=10)
    for address_line in vendor["address"]:
        pdf.cell(0, 5, address_line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    meta = [("N° de facture / Invoice no.", inv["invoice_number"]),
            ("Date de facture / Date", fmt_date(inv["invoice_date"], vendor["dates"])),
            ("Votre commande / Your order", inv["po_number"])]
    for label, value in meta:
        pdf.cell(60, 5, label + " :")
        pdf.cell(0, 5, value, new_x="LMARGIN", new_y="NEXT")
    bill_to(pdf, "Client")
    with pdf.table(col_widths=(24, 82, 14, 24, 26),
                   text_align=("LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT"),
                   borders_layout="HORIZONTAL_LINES") as table:
        table.row(["Réf.", "Désignation / Description", "Qté", "P.U. HT", "Montant HT"])
        for line in inv["lines"]:
            table.row([line["sku"], line["description"], *money_row(vendor, line)])


def ledger(pdf: FPDF, inv: dict, vendor: dict) -> None:
    """Typewriter style: Courier, numbered lines, all meta on one row."""
    pdf.footer_font = "Courier"
    pdf.set_font("Courier", "B", 14)
    pdf.cell(0, 7, vendor["name"].upper(), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", size=9)
    pdf.cell(0, 5, ", ".join(vendor["address"]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Courier", "B", 9)
    pdf.cell(0, 5, f"INVOICE NO. {inv['invoice_number']}    "
                   f"DATE {fmt_date(inv['invoice_date'], vendor['dates'])}    "
                   f"CUST. ORDER NO. {inv['po_number']}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Courier", size=9)
    bill_to(pdf, "Sold to")
    with pdf.table(col_widths=(8, 22, 76, 14, 24, 26),
                   text_align=("RIGHT", "LEFT", "LEFT", "RIGHT", "RIGHT", "RIGHT"),
                   borders_layout="SINGLE_TOP_LINE") as table:
        table.row(["#", "Item", "Description", "Qty", "Unit", "Ext."])
        for n, line in enumerate(inv["lines"], start=1):
            table.row([str(n), line["sku"], line["description"], *money_row(vendor, line)])


LAYOUTS = {"classic": classic, "band": band, "european": european, "ledger": ledger}

TOTAL_LABELS = {
    "classic": ("Subtotal", "Sales tax ({rate}%)", "Total"),
    "band": ("Net amount", "Sales tax {rate}%", "Amount due"),
    "european": ("Total HT", "TVA {rate} %", "Total TTC"),
    "ledger": ("Merchandise", "Tax @ {rate}%", "BALANCE DUE"),
}


def bill_to(pdf: FPDF, label: str) -> None:
    pdf.ln(4)
    pdf.cell(0, 5, label + ":", new_x="LMARGIN", new_y="NEXT")
    for buyer_line in BUYER:
        pdf.cell(0, 5, "    " + buyer_line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)


def totals_and_note(pdf: FPDF, inv: dict, vendor: dict, note: str) -> None:
    """Totals block under the table (kept on one page), then the free-text note."""
    if pdf.get_y() > pdf.h - 55:
        pdf.add_page()
    pdf.ln(3)
    labels = TOTAL_LABELS[vendor["layout"]]
    values = (inv["subtotal"], inv["tax"], inv["total"])
    for label, value in zip(labels, values, strict=True):
        pdf.cell(130, 6, "")
        pdf.cell(25, 6, label.format(rate=fmt_rate(vendor["tax_rate"])))
        pdf.cell(0, 6, fmt_money(value, vendor["money"]), align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.multi_cell(0, 5, "Notes: " + note)


def scan(pdf: bytes, rng: random.Random) -> bytes:
    """Make a digital PDF look scanned: image-only pages, tilted 1-2 degrees, blurred or
    speckled, a little faded. `rng` drives every random choice, so output is reproducible."""
    pages = []
    for page in pdfium.PdfDocument(pdf):
        img = page.render(scale=SCAN_DPI / 72).to_pil().convert("L")
        angle = rng.uniform(1.0, 2.0) * rng.choice([-1, 1])
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
        if rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 0.9)))
        else:
            img = speckle(img, rng)
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.75, 0.9))
        pages.append(img)
    out = io.BytesIO()
    fixed = FIXED_DATE.utctimetuple()  # Pillow stamps the current time unless told otherwise
    pages[0].save(out, "PDF", save_all=True, append_images=pages[1:], resolution=SCAN_DPI,
                  quality=70, creationDate=fixed, modDate=fixed)
    return out.getvalue()


def speckle(img: Image.Image, rng: random.Random) -> Image.Image:
    """Sprinkle ~0.4% black dots. Noise comes from `rng` (Pillow's own noise isn't seedable)."""
    noise = Image.frombytes("L", img.size, rng.randbytes(img.width * img.height))
    dots = noise.point(lambda v: 0 if v == 0 else 255)
    return ImageChops.darker(img, dots)

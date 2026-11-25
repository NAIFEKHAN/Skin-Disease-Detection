"""
PDF Report Generation Service for Skin Disease Detection.
Uses ReportLab to generate publication-quality, clinical decision-support summary reports
with multi-modal image matrices, educational disease overviews, cost breakdowns, and disclaimers.

Template Version: 2.4 (A4 Portrait, Unicode INR Glyph, Grouped Horizontal Cost Chart, 2x2 Image Grid)
"""

import os
import io
import json
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
    PageBreak
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

PDF_TEMPLATE_VERSION = "2.4"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
BASE_DIR = Path(__file__).resolve().parent.parent
DISEASE_INFO_PATH = BASE_DIR / "data" / "disease_info.json"

# ==============================================================================
# FONT REGISTRATION (TrueType Unicode Support for ₹ U+20B9)
# ==============================================================================
FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
HAS_RUPEE_FONT = False

FONT_SEARCH_PATHS = [
    (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf", "SegoeUI", "SegoeUI-Bold"),
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf", "Arial", "Arial-Bold"),
    (r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\calibrib.ttf", "Calibri", "Calibri-Bold"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans", "DejaVuSans-Bold")
]

for r_path, b_path, r_name, b_name in FONT_SEARCH_PATHS:
    if os.path.exists(r_path) and os.path.exists(b_path):
        try:
            pdfmetrics.registerFont(TTFont(r_name, r_path))
            pdfmetrics.registerFont(TTFont(b_name, b_path))
            FONT_REGULAR = r_name
            FONT_BOLD = b_name
            HAS_RUPEE_FONT = True
            break
        except Exception:
            continue

# Configure matching font families in Matplotlib
plt.rcParams["font.sans-serif"] = [FONT_REGULAR, "Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def format_inr(value: Union[int, float]) -> str:
    """Formats an integer or float amount into standard Indian currency representation."""
    try:
        val_int = int(round(float(value)))
        if HAS_RUPEE_FONT:
            return f"₹{val_int:,}"
        return f"INR {val_int:,}"
    except (ValueError, TypeError):
        return str(value)


def format_inr_range(min_val: Union[int, float], max_val: Union[int, float]) -> str:
    """Formats minimum and maximum amounts into a clean Indian currency range."""
    try:
        min_int = int(round(float(min_val)))
        max_int = int(round(float(max_val)))
        if HAS_RUPEE_FONT:
            return f"₹{min_int:,} – ₹{max_int:,}"
        return f"INR {min_int:,} – {max_int:,}"
    except (ValueError, TypeError):
        return f"{min_val} – {max_val}"


# ==============================================================================
# MATPLOTLIB STATIC CHART GENERATORS FOR PDF
# ==============================================================================

def generate_pdf_prediction_chart(top3_items, width=6.2, height=1.45) -> Optional[io.BytesIO]:
    """
    Generates horizontal bar chart for top-3 differential predictions.
    Displays complete wrapped labels, exact percentage scores, and Sky Blue/Orange/Light Blue bars.
    """
    if not top3_items:
        return None
    try:
        labels = []
        values = []
        for item in reversed(top3_items):
            name = item.get("display_name") or item.get("disease_name") or item.get("class_name") or item.get("class_code") or ""
            code = item.get("class_code")
            full_label = f"{name} ({code})" if code and code not in name else name
            wrapped_label = textwrap.fill(full_label, width=28)
            labels.append(wrapped_label)

            p = item.get("percentage")
            if isinstance(p, (int, float)):
                values.append(float(p))
            elif isinstance(p, str) and "%" in p:
                values.append(float(p.replace("%", "").strip()))
            else:
                prob = item.get("probability", 0.0)
                values.append(float(prob) * 100)

        # Palette: #00AFE0 (Sky Blue), #FF8500 (Orange), #AEE2F5 (Light Blue)
        palette_map = {
            3: ["#AEE2F5", "#FF8500", "#00AFE0"],
            2: ["#FF8500", "#00AFE0"],
            1: ["#00AFE0"]
        }
        bar_colors = palette_map.get(len(labels), ["#00AFE0"] * len(labels))

        fig, ax = plt.subplots(figsize=(width, height), dpi=180)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        bars = ax.barh(labels, values, color=bar_colors, edgecolor="#E5E7EB", height=0.55)

        for bar, val in zip(bars, values):
            ax.text(
                val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", ha="left", fontsize=8.5, fontweight="bold", color="#1F2937"
            )

        max_v = max(values) if values else 100
        ax.set_xlim(0, min(100, max(15, max_v + 16)))
        ax.set_xlabel("Model Score (%)", fontsize=8, color="#4B5563", labelpad=3)
        ax.tick_params(axis="y", labelsize=8.5, colors="#1F2937")
        ax.tick_params(axis="x", labelsize=8, colors="#6B7280")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D1D5DB")
        ax.spines["bottom"].set_color("#D1D5DB")
        ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="#E5E7EB")
        plt.tight_layout(pad=0.35)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=False)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Error generating prediction chart: {e}")
        return None


def generate_pdf_lesion_doughnut(lesion_area_pct: float, width=1.8, height=1.8) -> Optional[io.BytesIO]:
    """
    Generates doughnut chart for lesion image coverage.
    Maintains perfect circular 1:1 aspect ratio and centered percentage value.
    """
    try:
        lesion_pct = max(0.0, min(100.0, float(lesion_area_pct)))
        skin_pct = max(0.0, 100.0 - lesion_pct)

        fig, ax = plt.subplots(figsize=(width, height), dpi=180)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        colors_list = ["#FF8500", "#AEE2F5"]
        ax.pie(
            [lesion_pct, skin_pct],
            colors=colors_list,
            startangle=90,
            wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5)
        )
        ax.text(0, 0.06, f"{lesion_pct:.1f}%", ha="center", va="center", fontsize=11, fontweight="bold", color="#FF8500")
        ax.text(0, -0.22, "Coverage", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#6B7280")

        ax.axis("equal")
        plt.tight_layout(pad=0.15)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=False)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Error generating lesion doughnut chart: {e}")
        return None


def generate_pdf_cost_chart(breakdown_items, width=6.2, height=1.85) -> Optional[io.BytesIO]:
    """
    Generates grouped horizontal bar chart comparing minimum and maximum treatment costs across 5 tiers.
    Horizontal bars prevent category label truncation or rotation and display clean INR values.
    """
    if not breakdown_items:
        return None
    try:
        tiers = []
        min_costs = []
        max_costs = []

        label_mapping = {
            "doctor consultation": "Doctor Consultation",
            "diagnostic tests": "Diagnostic Tests",
            "diagnostic testing": "Diagnostic Tests",
            "medication": "Medication & Care",
            "medication & dressings": "Medication & Care",
            "treatment or procedure": "Procedure / Intervention",
            "procedure / intervention": "Procedure / Intervention",
            "treatment & procedures": "Procedure / Intervention",
            "treatment": "Procedure / Intervention",
            "follow-up care": "Follow-up Consultation",
            "follow-up consultation": "Follow-up Consultation",
            "follow-up": "Follow-up Consultation"
        }

        # Reverse so Consultation appears on top
        for item in reversed(breakdown_items):
            cat = (item.get("category") or item.get("category_name") or "Tier").strip()
            clean_cat = label_mapping.get(cat.lower(), cat)
            tiers.append(clean_cat)

            min_c = item.get("min_cost") or item.get("cost_range_inr", {}).get("min", 0)
            max_c = item.get("max_cost") or item.get("cost_range_inr", {}).get("max", 0)
            min_costs.append(float(min_c))
            max_costs.append(float(max_c))

        y = np.arange(len(tiers))
        bar_h = 0.35

        fig, ax = plt.subplots(figsize=(width, height), dpi=180)
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        ax.barh(y + bar_h/2, min_costs, bar_h, label="Minimum Cost (INR)", color="#00AFE0", edgecolor="#E5E7EB")
        ax.barh(y - bar_h/2, max_costs, bar_h, label="Maximum Cost (INR)", color="#FF8500", edgecolor="#E5E7EB")

        currency_prefix = "₹" if HAS_RUPEE_FONT else "INR "
        ax.set_yticks(y)
        ax.set_yticklabels(tiers, fontsize=8, fontweight="bold", color="#1F2937")
        ax.set_xlabel(f"Estimated Cost in Indian Rupees ({currency_prefix})", fontsize=7.5, color="#4B5563", labelpad=3)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{currency_prefix}{int(x):,}"))
        ax.tick_params(axis="x", labelsize=7.5, colors="#6B7280")
        ax.xaxis.grid(True, linestyle="--", alpha=0.5, color="#E5E7EB")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D1D5DB")
        ax.spines["bottom"].set_color("#D1D5DB")

        ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor="#D1D5DB", fontsize=7.5)
        plt.tight_layout(pad=0.35)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor(), transparent=False)
        plt.close(fig)
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"Error generating cost comparison chart: {e}")
        return None


# ==============================================================================
# TWO-PASS CANVAS FOR DYNAMIC PAGE NUMBERING & CLEAN RUNNING FOOTER
# ==============================================================================

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and draw total page count
    and clean running footer across all A4 pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        self.saveState()
        self.setFont(FONT_REGULAR, 8)
        self.setFillColor(colors.HexColor("#6B7280"))

        page_w, page_h = self._pagesize

        # Running footer separator line
        self.setStrokeColor(colors.HexColor("#E5E7EB"))
        self.setLineWidth(0.6)
        self.line(46, 30, page_w - 46, 30)

        # Left running footer
        self.drawString(46, 18, "Skin Disease Detection • Educational Preliminary Screening Prototype")

        # Right running footer
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(page_w - 46, 18, page_str)

        self.restoreState()


# ==============================================================================
# MAIN REPORT GENERATOR SERVICE
# ==============================================================================

class ReportGenerator:
    """Generates professional, printable A4 PDF reports for skin lesion analyses."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir if output_dir else REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._disease_cache: Optional[Dict[str, Any]] = None

    def _get_disease_info(self, code: str) -> Optional[Dict[str, Any]]:
        """Loads educational disease info from local JSON."""
        if self._disease_cache is None:
            if DISEASE_INFO_PATH.exists():
                try:
                    with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._disease_cache = {
                            d["code"].upper(): d for d in data.get("diseases", [])
                        }
                except Exception:
                    self._disease_cache = {}
            else:
                self._disease_cache = {}
        return self._disease_cache.get(code.upper())

    def generate_report(self, record_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Builds a comprehensive, publication-quality A4 PDF report from analysis record data.
        Returns the absolute filepath to the generated PDF.
        """
        ref_id = record_data.get("reference_id", "REF-UNKNOWN")
        if not filename:
            filename = f"report_{ref_id}.pdf"

        pdf_path = self.output_dir / filename

        # A4 portrait: 595.27 x 841.89 pt
        # Consistent 16.2 mm margins (46 pt): printable width = 503.27 pt
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            leftMargin=46,
            rightMargin=46,
            topMargin=40,
            bottomMargin=44
        )
        doc.title = f"Skin Disease Detection Report — {ref_id}"
        doc.author = "Skin Disease Detection System"
        doc.subject = f"Template v{PDF_TEMPLATE_VERSION}"

        styles = getSampleStyleSheet()

        # Forest green healthcare theme palette
        primary_color = colors.HexColor("#14532D")
        accent_color = colors.HexColor("#228B22")
        dark_text = colors.HexColor("#1F2937")
        muted_text = colors.HexColor("#4B5563")
        border_color = colors.HexColor("#D1D5DB")
        sec_bg = colors.HexColor("#F0FDF4")
        light_bg = colors.HexColor("#F9FAFB")
        alert_bg = colors.HexColor("#FEF2F2")
        alert_border = colors.HexColor("#FCA5A5")

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=15,
            leading=18,
            textColor=primary_color,
            fontName=FONT_BOLD,
            spaceAfter=2
        )

        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=11,
            textColor=muted_text,
            fontName=FONT_REGULAR,
            spaceAfter=4
        )

        section_heading = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=10,
            leading=12.5,
            textColor=primary_color,
            fontName=FONT_BOLD,
            spaceBefore=6,
            spaceAfter=3,
            keepWithNext=True
        )

        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=8,
            leading=10.5,
            textColor=dark_text,
            fontName=FONT_REGULAR
        )

        body_bold = ParagraphStyle(
            "BodyBold",
            parent=body_style,
            fontName=FONT_BOLD
        )

        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=7.2,
            leading=9.5,
            textColor=muted_text,
            fontName=FONT_REGULAR
        )

        elements = []

        # ======================================================================
        # PAGE 1 — ANALYSIS RESULTS
        # ======================================================================

        # 1. Header Banner
        elements.append(Paragraph("Skin Disease Detection — Analysis Report", title_style))
        elements.append(Paragraph("Decision-Support Screening Prototype • ResNet18 U-Net & Vision Transformer ViT-B/16", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=5))

        # 2. Metadata Section Table
        meta_data = [
            [
                Paragraph(f"<b>Report Reference:</b> <font color='#14532D'>{ref_id}</font>", body_style),
                Paragraph(f"<b>Date & Time:</b> {record_data.get('created_at', 'N/A')}", body_style),
                Paragraph("<b>Screening Scope:</b> Preliminary Academic Prototype", body_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[167, 168, 168])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), sec_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 5))

        # 3. Model Prediction Summary & Stat Findings
        elements.append(Paragraph("1. Model Prediction & Clinical Concern Indicator", section_heading))

        pred_name = record_data.get("predicted_disease_name") or record_data.get("predicted_disease") or "Unknown"
        pred_code = (record_data.get("predicted_class_code") or record_data.get("predicted_class") or "").upper()

        conf_val = record_data.get("confidence_percentage")
        if conf_val is None:
            c_score = record_data.get("confidence_score", record_data.get("confidence", 0.0))
            conf_val = f"{float(c_score) * 100:.1f}%" if c_score else "N/A"
        elif isinstance(conf_val, (int, float)):
            conf_val = f"{conf_val:.1f}%"

        area_val = record_data.get("relative_lesion_area") or record_data.get("lesion_area_percentage") or 0.0
        lesion_area = f"{float(area_val):.2f}%"

        sev_data = record_data.get("severity") or record_data.get("severity_details") or {}
        if isinstance(sev_data, str):
            sev_level = sev_data
        else:
            sev_level = sev_data.get("level") or record_data.get("severity_grade") or "Moderate Concern"

        findings_data = [
            [
                Paragraph("<b>Predicted Lesion Category:</b>", body_style),
                Paragraph(f"<b><font color='#14532D'>{pred_name}</font></b> ({pred_code})", body_style),
                Paragraph("<b>Model Confidence:</b>", body_style),
                Paragraph(f"<b>{conf_val}</b>", body_style)
            ],
            [
                Paragraph("<b>Prototype Concern Indicator:</b>", body_style),
                Paragraph(f"<b>{sev_level}</b>", body_style),
                Paragraph("<b>Relative Image Coverage:</b>", body_style),
                Paragraph(f"{lesion_area} of photo", body_style)
            ]
        ]

        findings_table = Table(findings_data, colWidths=[130, 150, 110, 113])
        findings_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
            ("PADDING", (0, 0), (-1, -1), 3.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
        ]))
        elements.append(findings_table)
        elements.append(Spacer(1, 5))

        # 4. Top-3 Disease Predictions with Matplotlib Chart & Table
        top3 = record_data.get("top_3_predictions") or record_data.get("top_3") or []
        if top3:
            elements.append(Paragraph("2. Top-3 Disease Predictions (Vision Transformer ViT-B/16)", section_heading))

            pred_chart_buf = generate_pdf_prediction_chart(top3, width=6.2, height=1.3)
            if pred_chart_buf:
                elements.append(RLImage(pred_chart_buf, width=503, height=92))
                elements.append(Spacer(1, 3))

            diff_rows = [
                [
                    Paragraph("<b>Rank</b>", body_bold),
                    Paragraph("<b>Category Code</b>", body_bold),
                    Paragraph("<b>Disease Category Name</b>", body_bold),
                    Paragraph("<b>Model Probability</b>", body_bold)
                ]
            ]
            for idx, item in enumerate(top3, start=1):
                pct_str = item.get("percentage")
                if isinstance(pct_str, (int, float)):
                    pct_str = f"{pct_str:.2f}%"
                elif not pct_str:
                    prob = item.get("probability", 0.0)
                    pct_str = f"{float(prob) * 100:.2f}%"

                name_str = item.get("display_name") or item.get("disease_name") or item.get("class_name") or ""
                code_str = item.get("class_code") or ""
                rank_str = f"#{item.get('rank', idx)}"

                diff_rows.append([
                    Paragraph(rank_str, body_style),
                    Paragraph(code_str, body_style),
                    Paragraph(name_str, body_style),
                    Paragraph(str(pct_str), body_style)
                ])

            diff_table = Table(diff_rows, colWidths=[40, 75, 268, 120])
            diff_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), sec_bg),
                ("BOX", (0, 0), (-1, -1), 0.5, border_color),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
                ("PADDING", (0, 0), (-1, -1), 2.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
            ]))
            elements.append(diff_table)
            elements.append(Spacer(1, 5))

        # 5. Four Image Analysis Outputs in a 2x2 Grid
        elements.append(Paragraph("3. Multi-Modal Image Analysis (Segmentation & Explainable AI)", section_heading))

        def load_scaled_img(rel_or_abs_path: Optional[str], target_w=150, target_h=105):
            """Loads and scales an image preserving 4:3 dermoscopic aspect ratio."""
            if not rel_or_abs_path:
                return Paragraph("<font color='gray'>Not Generated</font>", body_style)
            p = Path(rel_or_abs_path)
            if not p.is_absolute():
                p = BASE_DIR / p
            if p.exists():
                try:
                    return RLImage(str(p), width=target_w, height=target_h)
                except Exception:
                    return Paragraph("<font color='gray'>Render Error</font>", body_style)
            return Paragraph("<font color='gray'>Image Not Found</font>", body_style)

        is_common_skin = bool(record_data.get("analysis_mode") == "common_skin_conditions")

        img_orig = load_scaled_img(record_data.get("image_path") or record_data.get("input_image"))
        img_mask = load_scaled_img(record_data.get("mask_path") or record_data.get("binary_mask"))
        img_overlay = load_scaled_img(record_data.get("overlay_path") or record_data.get("overlay_image"))
        img_attn = load_scaled_img(record_data.get("attention_path") or record_data.get("attention_map_path") or record_data.get("attention_map"))

        if is_common_skin:
            mask_label = "<b>(B) Segmentation (Bypassed)</b>"
            overlay_label = "<b>(C) Contour (Bypassed)</b>"
        else:
            mask_label = "<b>(B) Segmentation Mask (U-Net)</b>"
            overlay_label = "<b>(C) Boundary Contour Overlay</b>"

        # 2x2 Image Grid Layout
        grid_data = [
            [
                Paragraph("<b>(A) Original Image</b>", body_style),
                Paragraph(mask_label, body_style)
            ],
            [img_orig, img_mask],
            [
                Paragraph(overlay_label, body_style),
                Paragraph("<b>(D) AI Attention Map (ViT-B/16)</b>", body_style)
            ],
            [img_overlay, img_attn]
        ]

        grid_table = Table(grid_data, colWidths=[251.5, 251.5])
        grid_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BACKGROUND", (0, 0), (-1, 0), light_bg),
            ("BACKGROUND", (0, 2), (-1, 2), light_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
            ("PADDING", (0, 0), (-1, -1), 2.5)
        ]))
        elements.append(grid_table)

        # Explicit Page Break to ensure Page 1 contains Analysis Results
        elements.append(PageBreak())

        # ======================================================================
        # PAGE 2 — ADDITIONAL ANALYSIS & COST
        # ======================================================================

        # 6. Lesion Image Coverage Area (Doughnut Chart)
        if is_common_skin:
            elements.append(Paragraph("4. Lesion Coverage Area (Clinical Protocol)", section_heading))
            cov_text = Paragraph(
                "<b>Segmentation Protocol:</b> Dermoscopic U-Net lesion segmentation was deliberately bypassed.<br/>"
                "<b>Clinical Rationale:</b> The trained U-Net segmentation model is validated exclusively on dermoscopic magnifications (HAM10000). "
                "Applying it to ambient clinical photography without separate clinical boundary annotations is avoided to prevent invalid contour claims.<br/>"
                "<font color='#4B5563' size='7'>*Clinical Note: For common skin conditions, explainable AI is provided exclusively via ViT-B/16 Self-Attention Rollout maps.</font>",
                body_style
            )
            cov_table = Table([[cov_text]], colWidths=[503])
        else:
            elements.append(Paragraph("4. Lesion Image Coverage Area (U-Net Segmentation)", section_heading))
            doughnut_buf = generate_pdf_lesion_doughnut(area_val, width=1.6, height=1.6)
            cov_text = Paragraph(
                f"<b>Segmented Lesion Region:</b> <font color='#FF8500'><b>{lesion_area}</b></font> of photograph area.<br/>"
                f"<b>Surrounding Skin / Background:</b> <font color='#00AFE0'><b>{max(0.0, 100.0 - float(area_val)):.2f}%</b></font>.<br/>"
                f"<b>Segmentation Architecture:</b> ResNet18 U-Net (Threshold 0.70).<br/>"
                f"<font color='#4B5563' size='7'>*Clinical Note: Represents relative 2D image pixel coverage, not physical tumor depth, volume, or clinical staging.</font>",
                body_style
            )
            if doughnut_buf:
                doughnut_img = RLImage(doughnut_buf, width=110, height=110)
                cov_table = Table([[doughnut_img, cov_text]], colWidths=[125, 378])
            else:
                cov_table = Table([[cov_text]], colWidths=[503])
        cov_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), sec_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 3.5)
        ]))
        elements.append(cov_table)
        elements.append(Spacer(1, 5))

        # 7. Educational Disease Information Callout
        disease_info = self._get_disease_info(pred_code)
        if disease_info:
            elements.append(Paragraph("5. Educational Disease Information", section_heading))
            overview_text = disease_info.get("short_description", "")
            chars = disease_info.get("common_characteristics", [])
            chars_str = " • ".join(chars[:3]) if chars else ""
            diag_str = " • ".join(disease_info.get("diagnostic_approaches", [])[:2])
            treat_str = " • ".join(disease_info.get("treatment_approaches", [])[:2])

            info_data = [
                [
                    Paragraph("<b>Overview:</b>", body_bold),
                    Paragraph(overview_text, body_style)
                ],
                [
                    Paragraph("<b>Common Features:</b>", body_bold),
                    Paragraph(chars_str, body_style)
                ],
                [
                    Paragraph("<b>Clinical Evaluation:</b>", body_bold),
                    Paragraph(diag_str, body_style)
                ],
                [
                    Paragraph("<b>General Management:</b>", body_bold),
                    Paragraph(treat_str, body_style)
                ]
            ]
            info_table = Table(info_data, colWidths=[105, 398])
            info_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), sec_bg),
                ("BOX", (0, 0), (-1, -1), 0.5, border_color),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
                ("PADDING", (0, 0), (-1, -1), 3),
                ("VALIGN", (0, 0), (-1, -1), "TOP")
            ]))
            elements.append(info_table)
            elements.append(Spacer(1, 5))

        # 8. Treatment Cost Comparison & Itemized Breakdown (INR)
        elements.append(Paragraph("6. Treatment Cost Comparison & Itemized Breakdown (INR)", section_heading))

        cost_info = record_data.get("cost_estimate") or record_data.get("cost_breakdown") or {}
        if isinstance(cost_info, str):
            try:
                cost_info = json.loads(cost_info)
            except Exception:
                cost_info = {}

        breakdown_items = cost_info.get("breakdown") or []
        if not breakdown_items and cost_info.get("breakdown_by_category"):
            breakdown_items = list(cost_info["breakdown_by_category"].values())

        if breakdown_items:
            cost_chart_buf = generate_pdf_cost_chart(breakdown_items, width=6.2, height=1.7)
            if cost_chart_buf:
                elements.append(RLImage(cost_chart_buf, width=503, height=115))
                elements.append(Spacer(1, 3))

            cost_rows = [
                [
                    Paragraph("<b>Care Category</b>", body_bold),
                    Paragraph("<b>Scope & Benchmark Services Included</b>", body_bold),
                    Paragraph("<b>Estimated Range (INR)</b>", body_bold)
                ]
            ]
            for item in breakdown_items:
                cat_name = item.get("category") or item.get("category_name") or ""
                desc = item.get("description") or ""
                min_c = item.get("min_cost") or item.get("cost_range_inr", {}).get("min", 0)
                max_c = item.get("max_cost") or item.get("cost_range_inr", {}).get("max", 0)
                range_str = format_inr_range(min_c, max_c)

                cost_rows.append([
                    Paragraph(f"<b>{cat_name}</b>", body_style),
                    Paragraph(desc, body_style),
                    Paragraph(range_str, body_style)
                ])

            tot_min = cost_info.get("total_min") or sum((i.get("min_cost", 0) for i in breakdown_items))
            tot_max = cost_info.get("total_max") or sum((i.get("max_cost", 0) for i in breakdown_items))
            tot_range = format_inr_range(tot_min, tot_max)

            cost_rows.append([
                Paragraph("<b>Total Estimated Range:</b>", body_bold),
                Paragraph("<b>Comprehensive illustrative pathway across all 5 tiers</b>", body_style),
                Paragraph(f"<b>{tot_range}</b>", body_bold)
            ])

            cost_table = Table(cost_rows, colWidths=[130, 245, 128])
            cost_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), sec_bg),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#D8F3DC")),
                ("BOX", (0, 0), (-1, -1), 0.5, border_color),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
                ("PADDING", (0, 0), (-1, -1), 2.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
            ]))
            elements.append(cost_table)
            elements.append(Spacer(1, 5))

        # 9. Recommendation & Medical Disclaimer Notice
        rec_text = (
            "<b>RECOMMENDATION & MEDICAL DISCLAIMER:</b> "
            "For educational and research demonstration only. This application does NOT provide a clinical diagnosis, "
            "biopsy confirmation, or medical treatment prescription. On held-out dermoscopic evaluation, overall multiclass "
            "accuracy is <b>64.59%</b> and Melanoma recall is <b>47.73%</b>. Benign model predictions must NEVER be interpreted "
            "as confirmation that malignancy is absent. Always consult a licensed dermatologist for any suspicious, changing, or bleeding lesion. "
            "Cost figures are illustrative Indian benchmark price bands, not hospital quotations."
        )

        disc_table = Table([[Paragraph(rec_text, disclaimer_style)]], colWidths=[503])
        disc_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), light_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, border_color),
            ("PADDING", (0, 0), (-1, -1), 4.5)
        ]))
        elements.append(disc_table)

        # Build document with dynamic NumberedCanvas
        doc.build(elements, canvasmaker=NumberedCanvas)
        return str(pdf_path)


# Global singleton instance
_report_generator_instance: Optional[ReportGenerator] = None


def get_report_generator() -> ReportGenerator:
    """Returns singleton instance of ReportGenerator."""
    global _report_generator_instance
    if _report_generator_instance is None:
        _report_generator_instance = ReportGenerator()
    return _report_generator_instance

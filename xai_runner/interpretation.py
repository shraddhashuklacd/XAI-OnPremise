from pathlib import Path
import pandas as pd
from PIL import Image as PILImage
import glob, os, math
from datetime import datetime
from openai import OpenAI

from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet

os.environ["OPENAI_API_KEY"] = # enter open ai key

# Auto-detect latest outputs run folder
ROOT_OUTPUTS = Path(__file__).resolve().parent / "outputs"

def find_latest_run(outputs_root: Path) -> Path:
    subdirs = [d for d in outputs_root.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError("No run folders inside outputs/")
    return max(subdirs, key=lambda d: d.name)

LATEST_RUN = find_latest_run(ROOT_OUTPUTS)

OUT_DIR = LATEST_RUN / "driver_analysis"
OUT_DIR2 = LATEST_RUN / "shap"

PDF_NAME = "analysis_summary.pdf"
FIXED_TIMESTAMP = "© 2025 | Catalytics Datum. Pvt. Ltd."

# All SHAP summary images
IMAGE_FILES = [
    OUT_DIR2 / "classification_shap_summary_raw.png",
    OUT_DIR2 / "classification_shap_summary_minmax.png",
    OUT_DIR2 / "regression_shap_summary_raw.png",
    OUT_DIR2 / "regression_shap_summary_minmax.png",
]

# Map of all required CSVs
CSV_FILES = {
    # Classification SHAP aggregates
    "Classification SHAP Aggregate – Raw": OUT_DIR2 / "classification_shap_aggregate_raw.csv",
    "Classification SHAP Aggregate – MinMax": OUT_DIR2 / "classification_shap_aggregate_minmax.csv",

    # Regression SHAP aggregates
    "Regression SHAP Aggregate – Raw": OUT_DIR2 / "regression_shap_aggregate_raw.csv",
    "Regression SHAP Aggregate – MinMax": OUT_DIR2 / "regression_shap_aggregate_minmax.csv",

    # Regression coefficients
    "Regression Coefficients – Raw": OUT_DIR / "regression_linear_coefficients.csv",
    "Regression Coefficients – MinMax": OUT_DIR / "regression_linear_coefficients_minmax.csv",

    # Classification coefficients
    "Classification Logistic Coefficients – Raw": OUT_DIR / "classification_logistic_coefficients.csv",
    "Classification Logistic Coefficients – MinMax": OUT_DIR / "classification_logistic_coefficients_minmax.csv",
}


# Optional: keep chart-level narrative for reuse (e.g., Power BI)
CHART_NARRATIVE_ROWS = []


def get_openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


CLIENT = get_openai_client()

def llm_panel_bullets(panel_title: str, global_context: str = "") -> list[str]:
    """Detailed, layman-language interpretation (3–5 bullets) for a chart/panel."""
    if CLIENT is None:
        return []
    try:
        system = (
            "You are a senior analytics lead. Explain charts in plain business terms for executives. "
            "Avoid heavy stats jargon; state direction, approximate magnitude, and a practical takeaway. "
            "Each bullet ≤ 20 words."
        )
        user = (
            f"Chart/panel: {panel_title}\n\n"
            "Context (may be partial stats table samples):\n"
            f"{global_context}\n\n"
            "Write 3–5 clear bullets: what it shows, notable pattern/anomaly, and an actionable next step."
        )
        resp = CLIENT.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        text = resp.output_text.strip()
        bullets = [b.strip("•-* ").strip() for b in text.splitlines() if b.strip()]
        if len(bullets) < 3:
            bullets += ["Observation needs more data to confirm.", "Monitor trend after next refresh."]
        return bullets[:5]
    except Exception:
        return []


def llm_row_one_liner(kind: str, row_dict: dict) -> str:

    def effect_word(val, bands):
        try:
            v = abs(float(val))
        except Exception:
            return None
        for thr, label in bands:
            if v >= thr:
                return label
        return "negligible"

    def is_sig(rd):
        p = rd.get("p") or rd.get("p_value") or rd.get("p-value") or rd.get("P")
        fdr = rd.get("FDR") or rd.get("fdr")
        try:
            return (p is not None and float(p) < 0.05) or (fdr is not None and float(fdr) < 0.05)
        except Exception:
            return False

    def name_of_var(rd):
        # for SHAP + regression tables
        for k in [
            "variable",
            "Variable",
            "feature",
            "Feature",
            "name",
            "Name",
            "predictor",
            "Predictor",
            "column",
            "original_variable",
            "encoded_variable",
        ]:
        
            if k in rd:
                return str(rd[k])
        return "Variable"

    def heuristic_anova(rd):
        var = name_of_var(rd)
        eta2 = rd.get("eta2") or rd.get("η²") or rd.get("eta_sq") or rd.get("eta^2")
        size = None
        try:
            if eta2 is not None:
                e = float(eta2)
                if e >= 0.14:
                    size = "large"
                elif e >= 0.06:
                    size = "medium"
                elif e >= 0.01:
                    size = "small"
                else:
                    size = "negligible"
        except Exception:
            pass
        if is_sig(rd) and size:
            return f"{var}: statistically significant ({size} effect). Prioritize for decisions."
        if is_sig(rd):
            return f"{var}: statistically significant impact. Investigate practical drivers."
        return f"{var}: no clear impact; lower priority."

    def heuristic_pearson(rd):
        var = name_of_var(rd)
        r = rd.get("r") or rd.get("R") or rd.get("pearson_r")
        try:
            r = float(r)
            direction = "positive" if r >= 0 else "negative"
            mag = effect_word(r, [(0.5, "strong"), (0.3, "moderate"), (0.1, "weak")])
            sig_txt = "significant" if is_sig(rd) else "not significant"
            return f"{var}: {mag} {direction} relationship ({sig_txt})."
        except Exception:
            return f"{var}: relationship unclear."

    def heuristic_chisq(rd):
        var = name_of_var(rd)
        v = (
            rd.get("cramers_v")
            or rd.get("Cramér’s V")
            or rd.get("cramers v")
            or rd.get("V")
            or rd.get("v")
        )
        mag = effect_word(v, [(0.5, "strong"), (0.3, "moderate"), (0.1, "weak")])
        sig_txt = "significant" if is_sig(rd) else "not significant"
        return f"{var}: {mag} association between categories ({sig_txt})."

    def heuristic_pointbiserial(rd):
        var = name_of_var(rd)
        r = rd.get("r") or rd.get("R") or rd.get("point_biserial_r") or rd.get("point-biserial r")
        try:
            r = float(r)
            direction = "higher for class=1" if r >= 0 else "higher for class=0"
            mag = effect_word(r, [(0.5, "strong"), (0.3, "moderate"), (0.1, "weak")])
            sig_txt = "significant" if is_sig(rd) else "not significant"
            return f"{var}: {mag} difference; {direction} ({sig_txt})."
        except Exception:
            return f"{var}: class difference unclear."

    def heuristic_shap(rd):

        var = rd.get("column") or name_of_var(rd)
        abs_mean = rd.get("absolute_mean")
        overall_mean = rd.get("overall_mean")
        pct_pos = rd.get("pct_positive")
        pct_neg = rd.get("pct_negative")

        try:
            abs_mean_f = float(abs_mean) if abs_mean is not None else 0.0
        except Exception:
            abs_mean_f = 0.0
        try:
            overall_mean_f = float(overall_mean) if overall_mean is not None else 0.0
        except Exception:
            overall_mean_f = 0.0
        try:
            pct_pos_f = float(pct_pos) if pct_pos is not None else 0.0
        except Exception:
            pct_pos_f = 0.0
        try:
            pct_neg_f = float(pct_neg) if pct_neg is not None else 0.0
        except Exception:
            pct_neg_f = 0.0

        mag = effect_word(abs_mean_f, [(0.20, "very strong"), (0.10, "strong"), (0.05, "moderate")])
        mag = mag or "weak"

        if overall_mean_f > 0:
            direction = "drives the prediction upwards when it increases"
        elif overall_mean_f < 0:
            direction = "pushes the prediction downwards when it increases"
        else:
            direction = "has a mixed impact on the prediction"

        if pct_pos_f >= pct_neg_f:
            dominance = "mostly positive contribution across observations"
        else:
            dominance = "mostly negative contribution across observations"

        return f"{var}: {mag} SHAP impact; {direction}, with {dominance}."

    def heuristic_linreg(rd):

        var = rd.get("original_variable") or rd.get("encoded_variable") or name_of_var(rd)
        coef = rd.get("coef")
        pval = rd.get("p_value") or rd.get("p-value") or rd.get("p")
        elas = rd.get("elasticity_at_mean")

        try:
            coef_f = float(coef) if coef is not None else 0.0
        except Exception:
            coef_f = 0.0
        try:
            elas_f = float(elas) if elas is not None else 0.0
        except Exception:
            elas_f = 0.0
        sig = False
        try:
            if pval is not None and float(pval) < 0.05:
                sig = True
        except Exception:
            sig = False

        direction = "positive" if coef_f > 0 else "negative" if coef_f < 0 else "neutral"
        mag = effect_word(elas_f, [(0.5, "strong"), (0.2, "moderate"), (0.05, "weak")]) or "very weak"
        sig_txt = "statistically significant" if sig else "not clearly significant"

        return f"{var}: {mag} {direction} effect on the outcome at mean levels, {sig_txt}."

    heuristics = {
        "anova": heuristic_anova,
        "pearson": heuristic_pearson,
        "chisq": heuristic_chisq,
        "pointbiserial": heuristic_pointbiserial,
        "shap": heuristic_shap,
        "linreg": heuristic_linreg,
    }

    if CLIENT is None:
        return heuristics[kind](row_dict)

    try:
        system = (
            "You write one-line, plain-English interpretations of statistical rows for executives. "
            "For SHAP tables, explain direction, strength, and whether positive/negative contributions dominate. "
            "For regression coefficients, mention sign, strength (elasticity/coefficient), and significance (p-value). "
            "Avoid jargon; one concise sentence."
        )
        user = f"Type={kind}. Summarize in one sentence for business leaders:\n{row_dict}"
        resp = CLIENT.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        line = resp.output_text.strip().splitlines()[0]
        return line
    except Exception:
        return heuristics[kind](row_dict)


def load_csvs(csv_map: dict[str, Path]) -> dict[str, pd.DataFrame]:
    dfs = {}
    for name, p in csv_map.items():
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                df = pd.read_csv(p, encoding="utf-8", engine="python")
            dfs[name] = df
    return dfs


def format_number(x):
    try:
        if pd.isna(x):
            return ""
        f = float(x)
        if math.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:.2f}"
    except Exception:
        return str(x)


def format_numeric_df_as_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(format_number)
    return out


def enrich_table_with_interpretation(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Adds 'Interpretation' one-liner per row using LLM (or heuristic)."""
    if df is None or df.empty:
        return df
    work = df.copy()
    lines = []
    for _, row in work.iterrows():
        lines.append(llm_row_one_liner(kind, row.to_dict()))
    work["Interpretation"] = lines
    return format_numeric_df_as_str(work)


def build_global_context_from_csvs(dfs_map: dict[str, pd.DataFrame], rows_per=8) -> str:
    parts = []
    for name, df in dfs_map.items():
        if df is None or df.empty:
            continue
        sample = df.head(rows_per)
        parts.append(
            f"### {name}\nColumns: {list(sample.columns)}\nHead:\n{sample.to_csv(index=False)}"
        )
    return "\n\n".join(parts) if parts else ""


def save_enriched_tables_for_powerbi(
    dfs_map: dict[str, pd.DataFrame],
    csv_paths: dict[str, Path],
) -> None:

    for name, path in csv_paths.items():
        df = dfs_map.get(name)
        if df is None or df.empty:
            continue

        lname = name.lower()
        if "shap aggregate" in lname:
            enriched = enrich_table_with_interpretation(df, "shap")
        elif "coefficients" in lname:
            enriched = enrich_table_with_interpretation(df, "linreg")
        else:
            enriched = format_numeric_df_as_str(df)

        # Overwrite the original file with the enriched version
        enriched.to_csv(path, index=False)

def estimate_col_width(series, header, font_name="Helvetica", font_size=8, padding=6):

    max_w = stringWidth(str(header), font_name, font_size)
    for v in series:
        txt = "" if pd.isna(v) else str(v)
        w = stringWidth(txt, font_name, font_size)
        if w > max_w:
            max_w = w
    return max_w + 2 * padding

def build_pdf_report(
    pdf_path: Path,
    images: list[Path],
    dfs_map: dict[str, pd.DataFrame],
    global_context: str,
):

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    title_style.textColor = colors.HexColor("#183A57")  
    h2_style = styles["Heading2"]
    h2_style.textColor = colors.HexColor("#183A57")
    normal = styles["Normal"]
    
    custom_page = (900, 600)  
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=custom_page,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    story = []

    # ---- Cover page ----
    story.append(Paragraph("Driver Analysis – Executive Summary", title_style))
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph("Auto-generated summary of key model drivers and statistics.", normal)
    )
    story.append(Spacer(1, 0.2 * inch))

    timestamp_text = (
        FIXED_TIMESTAMP
        or f"Auto-generated {datetime.now().strftime('%d %b %Y, %H:%M')}"
    )
    story.append(Paragraph(timestamp_text, normal))
    story.append(PageBreak())

    # ---- Chart pages (one image per page) ----
    for img_path in images:
        if not img_path.exists():
            continue

        title = img_path.stem.replace("_", " ").title()
        story.append(Paragraph(title, h2_style))
        story.append(Spacer(1, 0.15 * inch))

        # Fit image to a reasonable area
        max_w = 9.5 * inch
        max_h = 5.0 * inch
        with PILImage.open(img_path) as im:
            w_px, h_px = im.size
        aspect = w_px / float(h_px or 1)

        if (max_w / max_h) > aspect:
            # limited by height
            disp_h = max_h
            disp_w = disp_h * aspect
        else:
            # limited by width
            disp_w = max_w
            disp_h = disp_w / aspect

        story.append(RLImage(str(img_path), width=disp_w, height=disp_h))
        story.append(Spacer(1, 0.12 * inch))

        # Bullets using existing LLM helper
        bullets = llm_panel_bullets(title, global_context)

        # Persist bullets for reuse (e.g., Power BI)
        for i, b in enumerate(bullets, start=1):
            CHART_NARRATIVE_ROWS.append(
                {
                    "object_type": "chart",
                    "image_file": img_path.name,
                    "title": title,
                    "panel_index": 1,
                    "bullet_index": i,
                    "bullet_text": b,
                }
            )

        for b in bullets:
            story.append(Paragraph(f"• {b}", normal))

        story.append(PageBreak())

    # ---- Table pages (top rows + one-liner interpretations) ----
    for name, df in dfs_map.items():
        if df is None or df.empty:
            continue

        lname = name.lower()
        if "shap aggregate" in lname:
            enriched = enrich_table_with_interpretation(df, "shap")
            page_title = f"{name} – Key SHAP Drivers + One-liners"
        elif "coefficients" in lname:
            enriched = enrich_table_with_interpretation(df, "linreg")
            page_title = f"{name} – Coefficient Summary + One-liners"
        else:
            enriched = format_numeric_df_as_str(df)
            page_title = f"{name} – Top Rows"

        # Keep PDF light: top N rows
        sub = enriched.head(15).copy()

        # -------- ensure 'Interpretation' column is kept --------
        cols = list(sub.columns)
        if "Interpretation" in cols:
            # Put Interpretation at the end but guarantee it is included.
            other_cols = [c for c in cols if c != "Interpretation"]
            max_cols = 10  # total columns we allow on page
            # If too many, trim other columns but always keep Interpretation
            if len(other_cols) >= max_cols:
                other_cols = other_cols[: max_cols - 1]
            ordered_cols = other_cols + ["Interpretation"]
            sub = sub[ordered_cols]
        else:
            # old behaviour if there is no Interpretation column
            if sub.shape[1] > 10:
                sub = sub.iloc[:, :10]

        story.append(Paragraph(page_title, h2_style))
        story.append(Spacer(1, 0.12 * inch))

        table_rows = []
        header = list(sub.columns)
        table_rows.append(header)
        
        # Paragraph style for Interpretation
        interp_style = normal  
        
        for _, row in sub.iterrows():
            row_cells = []
            for col in sub.columns:
                val = "" if pd.isna(row[col]) else str(row[col])
                if col == "Interpretation":
                    # wrap Interpretation text
                    row_cells.append(Paragraph(val, interp_style))
                else:
                    row_cells.append(val)
            table_rows.append(row_cells)
        
        # ---- compute column widths ----
        PAGE_WIDTH, PAGE_HEIGHT = doc.pagesize
        available_width = PAGE_WIDTH - doc.leftMargin - doc.rightMargin
        
        cols = list(sub.columns)
        non_interp_cols = [c for c in cols if c != "Interpretation"]
        
        col_widths = []
        sum_non_interp = 0.0
        
        # 1) estimate natural widths for non-Interpretation columns
        for c in cols:
            if c == "Interpretation":
                col_widths.append(None)  # placeholder for now
            else:
                w = estimate_col_width(sub[c], c)
                col_widths.append(w)
                sum_non_interp += w
        
        # 2) remaining width for Interpretation
        remaining = available_width - sum_non_interp
        
        if "Interpretation" in cols:
            interp_idx = cols.index("Interpretation")
        
            # If remaining is too small or negative, scale all columns down proportionally
            if remaining <= 50:  # arbitrary minimum for Interpretation
                scale = available_width / sum_non_interp if sum_non_interp > 0 else 1.0
                for i, w in enumerate(col_widths):
                    if w is not None:
                        col_widths[i] = w * scale
                # give Interpretation a small share as well
                interp_width = max(available_width * 0.20, 60)  # 20% or at least 60pt
                col_widths[interp_idx] = interp_width
            else:
                # we have decent remaining width; give all of it to Interpretation
                col_widths[interp_idx] = remaining
        
        # 3) finally create the table with explicit widths
        tbl = Table(table_rows, colWidths=col_widths, repeatRows=1)


        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F6F8FB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0DBE9")),
                    ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )

        story.append(tbl)
        story.append(PageBreak())

    # Build the PDF
    doc.build(story)

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR2.mkdir(parents=True, exist_ok=True)

    images = [p for p in IMAGE_FILES if p.exists()]

    dfs = load_csvs(CSV_FILES)
    global_context = build_global_context_from_csvs(dfs, rows_per=8)

    save_enriched_tables_for_powerbi(dfs, CSV_FILES)

    pdf_path = LATEST_RUN / PDF_NAME
    build_pdf_report(pdf_path, images, dfs, global_context)
    print("Saved PDF:", pdf_path.resolve())

    if CHART_NARRATIVE_ROWS:
        narr_df = pd.DataFrame(CHART_NARRATIVE_ROWS)
        narr_csv_path = LATEST_RUN / "chart_narrative.csv"
        narr_df.to_csv(narr_csv_path, index=False)
        print("Saved chart-level narrative:", narr_csv_path.resolve())

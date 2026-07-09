import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import base64
from io import BytesIO
import html
import hashlib
import zipfile
from datetime import datetime, timezone
import math


# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="图片转 Visio 可编辑文件 V3",
    layout="wide"
)

st.title("图片转 Visio 可编辑文件 V3")
st.caption(
    "上传图片 → 自动识别线条/矩形/圆/文字 → 人工修正 → 导出原生可编辑 VSDX。"
)


# =========================
# 基础参数
# =========================

DEFAULT_COLUMNS = [
    "type",
    "x",
    "y",
    "w",
    "h",
    "text",
    "stroke",
    "fill",
    "stroke_width",
    "opacity",
    "font_size"
]


# =========================
# 基础工具函数
# =========================

def ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        df = pd.DataFrame(columns=DEFAULT_COLUMNS)

    for col in DEFAULT_COLUMNS:
        if col not in df.columns:
            if col == "type":
                df[col] = "rect"
            elif col == "text":
                df[col] = ""
            elif col == "stroke":
                df[col] = "#000000"
            elif col == "fill":
                df[col] = "none"
            elif col == "stroke_width":
                df[col] = 1
            elif col == "opacity":
                df[col] = 1.0
            elif col == "font_size":
                df[col] = 12
            else:
                df[col] = 0

    return df[DEFAULT_COLUMNS]


def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def uploaded_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def reset_recognition():
    st.session_state.pop("objects_df", None)


def safe_float(value, default=0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def xml_escape(value) -> str:
    return html.escape(str(value), quote=True)


def hex_to_rgb(hex_color):
    try:
        color = str(hex_color).strip()
        if not color.startswith("#"):
            return 0, 0, 0
        color = color.lstrip("#")
        if len(color) == 3:
            color = "".join([c * 2 for c in color])
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        return r, g, b
    except Exception:
        return 0, 0, 0


def color_cell(name, hex_color):
    r, g, b = hex_to_rgb(hex_color)
    return f'<Cell N="{name}" V="0" F="RGB({r},{g},{b})"/>'


def px_to_in(px, dpi):
    return float(px) / float(dpi)


# =========================
# 图像预处理
# =========================

def preprocess_image(image_np, enhance=True):
    if not enhance:
        return image_np

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


# =========================
# 识别矩形
# =========================

def detect_rectangles(
    image_np,
    min_area=500,
    max_area_ratio=0.95,
    stroke_color="#000000"
):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    img_h, img_w = gray.shape
    img_area = img_w * img_h

    objects = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < min_area:
            continue

        if area > img_area * max_area_ratio:
            continue

        if w < 10 or h < 10:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) >= 4:
            objects.append({
                "type": "rect",
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "text": "",
                "stroke": stroke_color,
                "fill": "none",
                "stroke_width": 1,
                "opacity": 1.0,
                "font_size": 12
            })

    return objects


# =========================
# 识别线条
# =========================

def detect_lines(
    image_np,
    min_line_length=40,
    max_line_gap=8,
    only_horizontal_vertical=True,
    angle_tolerance=5,
    stroke_color="#000000"
):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    objects = []

    if lines is None:
        return objects

    for line in lines:
        pts = np.array(line).reshape(-1)

        if len(pts) != 4:
            continue

        x1, y1, x2, y2 = pts

        dx = x2 - x1
        dy = y2 - y1

        length = np.sqrt(dx * dx + dy * dy)

        if length < min_line_length:
            continue

        if only_horizontal_vertical:
            angle = abs(np.degrees(np.arctan2(dy, dx)))
            is_horizontal = angle <= angle_tolerance or angle >= 180 - angle_tolerance
            is_vertical = abs(angle - 90) <= angle_tolerance

            if not (is_horizontal or is_vertical):
                continue

        objects.append({
            "type": "line",
            "x": int(x1),
            "y": int(y1),
            "w": int(dx),
            "h": int(dy),
            "text": "",
            "stroke": stroke_color,
            "fill": "none",
            "stroke_width": 1,
            "opacity": 1.0,
            "font_size": 12
        })

    return objects


# =========================
# 识别圆
# =========================

def detect_circles(
    image_np,
    min_radius=4,
    max_radius=30,
    stroke_color="#000000"
):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(8, min_radius * 2),
        param1=80,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    objects = []

    if circles is None:
        return objects

    circles = np.round(circles[0, :]).astype("int")

    for x, y, r in circles:
        objects.append({
            "type": "circle",
            "x": int(x - r),
            "y": int(y - r),
            "w": int(r * 2),
            "h": int(r * 2),
            "text": "",
            "stroke": stroke_color,
            "fill": "none",
            "stroke_width": 1,
            "opacity": 1.0,
            "font_size": 12
        })

    return objects


# =========================
# OCR 文字识别
# =========================

def detect_text(
    image: Image.Image,
    min_conf=80,
    stroke_color="#000000"
):
    objects = []

    try:
        data = pytesseract.image_to_data(
            image,
            lang="chi_sim+eng",
            config="--psm 11",
            output_type=pytesseract.Output.DATAFRAME
        )

        data = data.dropna()

        for _, row in data.iterrows():
            conf = safe_float(row.get("conf", -1), -1)

            if conf < min_conf:
                continue

            text = str(row.get("text", "")).strip()

            if not text:
                continue

            x = safe_int(row.get("left", 0))
            y = safe_int(row.get("top", 0))
            w = safe_int(row.get("width", 0))
            h = safe_int(row.get("height", 0))

            if w < 5 or h < 5:
                continue

            font_size = max(6, int(h * 0.9))

            objects.append({
                "type": "text",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "text": text,
                "stroke": stroke_color,
                "fill": "none",
                "stroke_width": 1,
                "opacity": 1.0,
                "font_size": font_size
            })

    except Exception as e:
        st.warning(f"OCR 文字识别失败，已跳过：{e}")

    return objects


# =========================
# SVG 预览
# =========================

def make_preview_svg(
    width,
    height,
    background_href,
    df,
    keep_background=True,
    background_opacity=1.0,
    show_edit_layer=True
):
    svg = []

    svg.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}px" height="{height}px" '
        f'viewBox="0 0 {width} {height}" version="1.1">'
    )

    if keep_background:
        svg.append(
            f'<image x="0" y="0" width="{width}" height="{height}" '
            f'xlink:href="{background_href}" '
            f'href="{background_href}" '
            f'opacity="{background_opacity}"/>'
        )

    if show_edit_layer:
        df = ensure_df(df)

        for _, row in df.iterrows():
            obj_type = str(row.get("type", "")).strip().lower()

            x = safe_int(row.get("x", 0))
            y = safe_int(row.get("y", 0))
            w = safe_int(row.get("w", 0))
            h = safe_int(row.get("h", 0))

            stroke = str(row.get("stroke", "#000000")).strip()
            fill = str(row.get("fill", "none")).strip()
            text = html.escape(str(row.get("text", "")))
            stroke_width = safe_float(row.get("stroke_width", 1), 1)
            opacity = safe_float(row.get("opacity", 1.0), 1.0)
            font_size = safe_int(row.get("font_size", 12), 12)

            if obj_type == "rect":
                if w <= 0 or h <= 0:
                    continue

                svg.append(
                    f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                    f'fill="{fill}" stroke="{stroke}" '
                    f'stroke-width="{stroke_width}" opacity="{opacity}"/>'
                )

            elif obj_type == "line":
                svg.append(
                    f'<line x1="{x}" y1="{y}" x2="{x + w}" y2="{y + h}" '
                    f'stroke="{stroke}" stroke-width="{stroke_width}" '
                    f'opacity="{opacity}"/>'
                )

            elif obj_type == "text":
                if not text:
                    continue

                svg.append(
                    f'<text x="{x}" y="{y + h}" '
                    f'font-size="{font_size}" '
                    f'font-family="Arial, Microsoft YaHei" '
                    f'fill="{stroke}" opacity="{opacity}">{text}</text>'
                )

            elif obj_type == "circle":
                r = max(1, min(abs(w), abs(h)) // 2)
                cx = x + r
                cy = y + r

                svg.append(
                    f'<circle cx="{cx}" cy="{cy}" r="{r}" '
                    f'fill="{fill}" stroke="{stroke}" '
                    f'stroke-width="{stroke_width}" opacity="{opacity}"/>'
                )

    svg.append("</svg>")
    return "\n".join(svg)


# =========================
# VSDX 生成：Shape XML
# =========================

def vsdx_shape_rect(shape_id, row, page_h_in, dpi):
    x = safe_float(row.get("x", 0))
    y = safe_float(row.get("y", 0))
    w = max(1, safe_float(row.get("w", 1)))
    h = max(1, safe_float(row.get("h", 1)))

    pin_x = px_to_in(x + w / 2, dpi)
    pin_y = page_h_in - px_to_in(y + h / 2, dpi)
    width = max(px_to_in(w, dpi), 0.01)
    height = max(px_to_in(h, dpi), 0.01)

    stroke = row.get("stroke", "#000000")
    stroke_width = max(px_to_in(safe_float(row.get("stroke_width", 1), 1), dpi), 0.003)

    return f'''
<Shape ID="{shape_id}" Type="Shape" LineStyle="0" FillStyle="0" TextStyle="0">
    <Cell N="PinX" V="{pin_x:.6f}"/>
    <Cell N="PinY" V="{pin_y:.6f}"/>
    <Cell N="Width" V="{width:.6f}"/>
    <Cell N="Height" V="{height:.6f}"/>
    <Cell N="LinePattern" V="1"/>
    <Cell N="FillPattern" V="0"/>
    <Cell N="LineWeight" V="{stroke_width:.6f}"/>
    {color_cell("LineColor", stroke)}
    <Section N="Geometry" IX="0">
        <Cell N="NoFill" V="1"/>
        <Cell N="NoLine" V="0"/>
        <Row T="MoveTo" IX="1">
            <Cell N="X" V="0"/>
            <Cell N="Y" V="0"/>
        </Row>
        <Row T="LineTo" IX="2">
            <Cell N="X" V="{width:.6f}"/>
            <Cell N="Y" V="0"/>
        </Row>
        <Row T="LineTo" IX="3">
            <Cell N="X" V="{width:.6f}"/>
            <Cell N="Y" V="{height:.6f}"/>
        </Row>
        <Row T="LineTo" IX="4">
            <Cell N="X" V="0"/>
            <Cell N="Y" V="{height:.6f}"/>
        </Row>
        <Row T="LineTo" IX="5">
            <Cell N="X" V="0"/>
            <Cell N="Y" V="0"/>
        </Row>
    </Section>
</Shape>
'''


def vsdx_shape_line(shape_id, row, page_h_in, dpi):
    x1_px = safe_float(row.get("x", 0))
    y1_px = safe_float(row.get("y", 0))
    x2_px = x1_px + safe_float(row.get("w", 0))
    y2_px = y1_px + safe_float(row.get("h", 0))

    x1 = px_to_in(x1_px, dpi)
    y1 = page_h_in - px_to_in(y1_px, dpi)
    x2 = px_to_in(x2_px, dpi)
    y2 = page_h_in - px_to_in(y2_px, dpi)

    left = min(x1, x2)
    right = max(x1, x2)
    bottom = min(y1, y2)
    top = max(y1, y2)

    width = max(right - left, 0.01)
    height = max(top - bottom, 0.01)

    pin_x = left + width / 2
    pin_y = bottom + height / 2

    local_x1 = 0 if x1 <= x2 else width
    local_x2 = width if x1 <= x2 else 0
    local_y1 = 0 if y1 <= y2 else height
    local_y2 = height if y1 <= y2 else 0

    stroke = row.get("stroke", "#000000")
    stroke_width = max(px_to_in(safe_float(row.get("stroke_width", 1), 1), dpi), 0.003)

    return f'''
<Shape ID="{shape_id}" Type="Shape" LineStyle="0" FillStyle="0" TextStyle="0">
    <Cell N="PinX" V="{pin_x:.6f}"/>
    <Cell N="PinY" V="{pin_y:.6f}"/>
    <Cell N="Width" V="{width:.6f}"/>
    <Cell N="Height" V="{height:.6f}"/>
    <Cell N="LinePattern" V="1"/>
    <Cell N="FillPattern" V="0"/>
    <Cell N="LineWeight" V="{stroke_width:.6f}"/>
    {color_cell("LineColor", stroke)}
    <Section N="Geometry" IX="0">
        <Cell N="NoFill" V="1"/>
        <Cell N="NoLine" V="0"/>
        <Row T="MoveTo" IX="1">
            <Cell N="X" V="{local_x1:.6f}"/>
            <Cell N="Y" V="{local_y1:.6f}"/>
        </Row>
        <Row T="LineTo" IX="2">
            <Cell N="X" V="{local_x2:.6f}"/>
            <Cell N="Y" V="{local_y2:.6f}"/>
        </Row>
    </Section>
</Shape>
'''


def vsdx_shape_circle(shape_id, row, page_h_in, dpi):
    x = safe_float(row.get("x", 0))
    y = safe_float(row.get("y", 0))
    w = max(1, safe_float(row.get("w", 1)))
    h = max(1, safe_float(row.get("h", 1)))

    pin_x = px_to_in(x + w / 2, dpi)
    pin_y = page_h_in - px_to_in(y + h / 2, dpi)

    width = max(px_to_in(w, dpi), 0.01)
    height = max(px_to_in(h, dpi), 0.01)

    stroke = row.get("stroke", "#000000")
    stroke_width = max(px_to_in(safe_float(row.get("stroke_width", 1), 1), dpi), 0.003)

    cx = width / 2
    cy = height / 2
    rx = width / 2
    ry = height / 2

    points = []
    steps = 24
    for i in range(steps + 1):
        angle = 2 * math.pi * i / steps
        px = cx + rx * math.cos(angle)
        py = cy + ry * math.sin(angle)
        points.append((px, py))

    rows = []
    rows.append(f'''
        <Row T="MoveTo" IX="1">
            <Cell N="X" V="{points[0][0]:.6f}"/>
            <Cell N="Y" V="{points[0][1]:.6f}"/>
        </Row>
    ''')

    for idx, (px, py) in enumerate(points[1:], start=2):
        rows.append(f'''
        <Row T="LineTo" IX="{idx}">
            <Cell N="X" V="{px:.6f}"/>
            <Cell N="Y" V="{py:.6f}"/>
        </Row>
        ''')

    geometry_rows = "\n".join(rows)

    return f'''
<Shape ID="{shape_id}" Type="Shape" LineStyle="0" FillStyle="0" TextStyle="0">
    <Cell N="PinX" V="{pin_x:.6f}"/>
    <Cell N="PinY" V="{pin_y:.6f}"/>
    <Cell N="Width" V="{width:.6f}"/>
    <Cell N="Height" V="{height:.6f}"/>
    <Cell N="LinePattern" V="1"/>
    <Cell N="FillPattern" V="0"/>
    <Cell N="LineWeight" V="{stroke_width:.6f}"/>
    {color_cell("LineColor", stroke)}
    <Section N="Geometry" IX="0">
        <Cell N="NoFill" V="1"/>
        <Cell N="NoLine" V="0"/>
        {geometry_rows}
    </Section>
</Shape>
'''


def vsdx_shape_text(shape_id, row, page_h_in, dpi):
    x = safe_float(row.get("x", 0))
    y = safe_float(row.get("y", 0))
    w = max(1, safe_float(row.get("w", 1)))
    h = max(1, safe_float(row.get("h", 1)))

    pin_x = px_to_in(x + w / 2, dpi)
    pin_y = page_h_in - px_to_in(y + h / 2, dpi)

    width = max(px_to_in(w, dpi), 0.05)
    height = max(px_to_in(h, dpi), 0.05)

    text = xml_escape(row.get("text", ""))
    stroke = row.get("stroke", "#000000")

    font_size_px = safe_float(row.get("font_size", 12), 12)
    font_size_in = max(font_size_px / 72.0, 0.08)

    return f'''
<Shape ID="{shape_id}" Type="Shape" LineStyle="0" FillStyle="0" TextStyle="0">
    <Cell N="PinX" V="{pin_x:.6f}"/>
    <Cell N="PinY" V="{pin_y:.6f}"/>
    <Cell N="Width" V="{width:.6f}"/>
    <Cell N="Height" V="{height:.6f}"/>
    <Cell N="LinePattern" V="0"/>
    <Cell N="FillPattern" V="0"/>
    <Text>{text}</Text>
    <Section N="Character" IX="0">
        <Row IX="0">
            <Cell N="Size" V="{font_size_in:.6f}"/>
            {color_cell("Color", stroke)}
        </Row>
    </Section>
</Shape>
'''


# =========================
# VSDX 包结构生成
# =========================

def make_native_editable_vsdx(df, img_width_px, img_height_px, dpi=96):
    df = ensure_df(df)

    page_w_in = max(px_to_in(img_width_px, dpi), 0.1)
    page_h_in = max(px_to_in(img_height_px, dpi), 0.1)

    shapes_xml = []
    shape_id = 1

    for _, row in df.iterrows():
        obj_type = str(row.get("type", "")).strip().lower()

        try:
            if obj_type == "rect":
                shapes_xml.append(vsdx_shape_rect(shape_id, row, page_h_in, dpi))
                shape_id += 1

            elif obj_type == "line":
                shapes_xml.append(vsdx_shape_line(shape_id, row, page_h_in, dpi))
                shape_id += 1

            elif obj_type == "circle":
                shapes_xml.append(vsdx_shape_circle(shape_id, row, page_h_in, dpi))
                shape_id += 1

            elif obj_type == "text":
                if str(row.get("text", "")).strip():
                    shapes_xml.append(vsdx_shape_text(shape_id, row, page_h_in, dpi))
                    shape_id += 1

        except Exception:
            continue

    shapes_xml_text = "\n".join(shapes_xml)

    content_types_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
    <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
    <Override PartName="/visio/document.xml" ContentType="application/vnd.ms-visio.drawing.main+xml"/>
    <Override PartName="/visio/pages/pages.xml" ContentType="application/vnd.ms-visio.pages+xml"/>
    <Override PartName="/visio/pages/page1.xml" ContentType="application/vnd.ms-visio.page+xml"/>
</Types>
'''

    rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/document" Target="visio/document.xml"/>
    <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
    <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
'''

    created_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    core_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties 
    xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" 
    xmlns:dc="http://purl.org/dc/elements/1.1/" 
    xmlns:dcterms="http://purl.org/dc/terms/" 
    xmlns:dcmitype="http://purl.org/dc/dcmitype/" 
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <dc:title>Image to Editable Visio</dc:title>
    <dc:creator>Streamlit Image to Visio Tool</dc:creator>
    <cp:lastModifiedBy>Streamlit Image to Visio Tool</cp:lastModifiedBy>
    <dcterms:created xsi:type="dcterms:W3CDTF">{created_time}</dcterms:created>
    <dcterms:modified xsi:type="dcterms:W3CDTF">{created_time}</dcterms:modified>
</cp:coreProperties>
'''

    app_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties 
    xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" 
    xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
    <Application>Microsoft Visio</Application>
</Properties>
'''

    document_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<VisioDocument 
    xmlns="http://schemas.microsoft.com/office/visio/2012/main" 
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" 
    xml:space="preserve">
    <DocumentProperties/>
    <DocumentSettings/>
    <Colors/>
    <FaceNames>
        <FaceName ID="0" Name="Arial" UnicodeRanges="-1" CharSets="0" Panos="020B0604020202020204"/>
    </FaceNames>
    <StyleSheets/>
    <DocumentSheet>
        <Cell N="DocLangID" V="1033"/>
    </DocumentSheet>
</VisioDocument>
'''

    document_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/pages" Target="pages/pages.xml"/>
</Relationships>
'''

    pages_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Pages 
    xmlns="http://schemas.microsoft.com/office/visio/2012/main" 
    xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
    <Page ID="0" NameU="Page-1" Name="Page-1">
        <Rel r:id="rId1"/>
    </Page>
</Pages>
'''

    pages_rels_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/page" Target="page1.xml"/>
</Relationships>
'''

    page1_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<PageContents 
    xmlns="http://schemas.microsoft.com/office/visio/2012/main" 
    xml:space="preserve">
    <PageSheet LineStyle="0" FillStyle="0" TextStyle="0">
        <Cell N="PageWidth" V="{page_w_in:.6f}"/>
        <Cell N="PageHeight" V="{page_h_in:.6f}"/>
        <Cell N="DrawingScale" V="1"/>
        <Cell N="PageScale" V="1"/>
        <Cell N="DrawingSizeType" V="0"/>
    </PageSheet>
    <Shapes>
        {shapes_xml_text}
    </Shapes>
</PageContents>
'''

    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("visio/document.xml", document_xml)
        zf.writestr("visio/_rels/document.xml.rels", document_rels_xml)
        zf.writestr("visio/pages/pages.xml", pages_xml)
        zf.writestr("visio/pages/_rels/pages.xml.rels", pages_rels_xml)
        zf.writestr("visio/pages/page1.xml", page1_xml)

    buffer.seek(0)
    return buffer.getvalue()


# =========================
# 导出 ZIP
# =========================

def make_export_zip(image, df, preview_svg, editable_vsdx_bytes, json_text):
    zip_buffer = BytesIO()

    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("original_image.png", img_bytes)
        zf.writestr("preview_with_background.svg", preview_svg.encode("utf-8"))
        zf.writestr("native_editable.vsdx", editable_vsdx_bytes)
        zf.writestr("recognition_result.json", json_text.encode("utf-8"))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# =========================
# 侧边栏参数
# =========================

st.sidebar.header("识别参数")

enable_enhance = st.sidebar.checkbox("识别前增强图片", value=True)

enable_rect = st.sidebar.checkbox("识别矩形", value=True)
enable_line = st.sidebar.checkbox("识别线条", value=True)
enable_circle = st.sidebar.checkbox("识别圆/焊球", value=False)
enable_text = st.sidebar.checkbox("识别文字 OCR", value=False)

st.sidebar.divider()

min_area = st.sidebar.slider(
    "矩形最小面积",
    min_value=100,
    max_value=5000,
    value=800,
    step=100
)

min_line_length = st.sidebar.slider(
    "线条最小长度",
    min_value=10,
    max_value=300,
    value=60,
    step=10
)

max_line_gap = st.sidebar.slider(
    "线条最大断点连接距离",
    min_value=1,
    max_value=30,
    value=8,
    step=1
)

only_hv = st.sidebar.checkbox(
    "只识别水平/垂直线",
    value=True
)

circle_min_r = st.sidebar.slider(
    "圆最小半径",
    min_value=2,
    max_value=30,
    value=4,
    step=1
)

circle_max_r = st.sidebar.slider(
    "圆最大半径",
    min_value=5,
    max_value=80,
    value=25,
    step=1
)

ocr_conf = st.sidebar.slider(
    "OCR 最低置信度",
    min_value=0,
    max_value=100,
    value=80,
    step=5
)

overlay_color = st.sidebar.color_picker(
    "识别图层颜色",
    value="#000000"
)

st.sidebar.divider()

export_dpi = st.sidebar.number_input(
    "导出 VSDX DPI 换算",
    min_value=72,
    max_value=300,
    value=96,
    step=1
)


# =========================
# 上传图片
# =========================

uploaded = st.file_uploader(
    "上传需要还原的图片",
    type=["png", "jpg", "jpeg", "bmp"]
)

if uploaded is None:
    st.info("请先上传图片。")
    st.stop()

file_bytes = uploaded.getvalue()
file_hash = uploaded_file_hash(file_bytes)

if st.session_state.get("file_hash") != file_hash:
    st.session_state["file_hash"] = file_hash
    reset_recognition()

image = Image.open(BytesIO(file_bytes)).convert("RGB")
image_np = np.array(image)

width, height = image.size

processed_np = preprocess_image(image_np, enhance=enable_enhance)


# =========================
# 主页面布局
# =========================

col1, col2 = st.columns(2)

with col1:
    st.subheader("原始图片")
    st.image(image, use_container_width=True)
    st.write(f"图片尺寸：{width} × {height} px")

with col2:
    st.subheader("识别控制")

    st.write(
        "说明：要得到可编辑 VSDX，必须导出识别出来的矢量对象。"
        "原图底图只能作为视觉参考，不能真正逐个编辑。"
    )

    run_recognition = st.button(
        "开始识别 / 重新识别",
        type="primary",
        use_container_width=True
    )

    clear_objects = st.button(
        "清空识别图层",
        use_container_width=True
    )

    if clear_objects:
        st.session_state["objects_df"] = pd.DataFrame(columns=DEFAULT_COLUMNS)
        st.success("已清空识别图层。")


# =========================
# 执行识别
# =========================

if run_recognition or "objects_df" not in st.session_state:
    objects = []

    with st.spinner("正在识别图片元素..."):
        if enable_rect:
            try:
                rects = detect_rectangles(
                    processed_np,
                    min_area=min_area,
                    stroke_color=overlay_color
                )
                objects += rects
                st.info(f"矩形识别完成：{len(rects)} 个")
            except Exception as e:
                st.warning(f"矩形识别失败，已跳过：{e}")

        if enable_line:
            try:
                lines = detect_lines(
                    processed_np,
                    min_line_length=min_line_length,
                    max_line_gap=max_line_gap,
                    only_horizontal_vertical=only_hv,
                    stroke_color=overlay_color
                )
                objects += lines
                st.info(f"线条识别完成：{len(lines)} 条")
            except Exception as e:
                st.warning(f"线条识别失败，已跳过：{e}")

        if enable_circle:
            try:
                circles = detect_circles(
                    processed_np,
                    min_radius=circle_min_r,
                    max_radius=circle_max_r,
                    stroke_color=overlay_color
                )
                objects += circles
                st.info(f"圆/焊球识别完成：{len(circles)} 个")
            except Exception as e:
                st.warning(f"圆识别失败，已跳过：{e}")

        if enable_text:
            try:
                texts = detect_text(
                    image,
                    min_conf=ocr_conf,
                    stroke_color=overlay_color
                )
                objects += texts
                st.info(f"文字识别完成：{len(texts)} 个")
            except Exception as e:
                st.warning(f"文字识别失败，已跳过：{e}")

    st.session_state["objects_df"] = ensure_df(pd.DataFrame(objects))


# =========================
# 人工修正表
# =========================

st.divider()

st.subheader("识别结果 / 人工修正")

objects_df = ensure_df(
    st.session_state.get(
        "objects_df",
        pd.DataFrame(columns=DEFAULT_COLUMNS)
    )
)

edited_df = st.data_editor(
    objects_df,
    num_rows="dynamic",
    use_container_width=True,
    height=360,
    column_config={
        "type": st.column_config.SelectboxColumn(
            "type",
            options=["rect", "line", "text", "circle"],
            required=True
        ),
        "x": st.column_config.NumberColumn("x", step=1),
        "y": st.column_config.NumberColumn("y", step=1),
        "w": st.column_config.NumberColumn("w", step=1),
        "h": st.column_config.NumberColumn("h", step=1),
        "text": st.column_config.TextColumn("text"),
        "stroke": st.column_config.TextColumn("stroke"),
        "fill": st.column_config.TextColumn("fill"),
        "stroke_width": st.column_config.NumberColumn(
            "stroke_width",
            min_value=0.1,
            step=0.5
        ),
        "opacity": st.column_config.NumberColumn(
            "opacity",
            min_value=0.0,
            max_value=1.0,
            step=0.05
        ),
        "font_size": st.column_config.NumberColumn(
            "font_size",
            min_value=1,
            step=1
        ),
    },
    key="object_editor"
)

st.session_state["objects_df"] = ensure_df(edited_df)


# =========================
# 预览 / 导出
# =========================

st.divider()

st.subheader("预览 / 导出")

preview_col1, preview_col2, preview_col3 = st.columns(3)

with preview_col1:
    keep_background = st.checkbox(
        "预览时显示原图底图",
        value=True
    )

with preview_col2:
    background_opacity = st.slider(
        "底图不透明度",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05
    )

with preview_col3:
    show_edit_layer = st.checkbox(
        "预览时显示识别图层",
        value=True
    )

preview_background_href = f"data:image/png;base64,{image_to_base64(image)}"

preview_svg = make_preview_svg(
    width=width,
    height=height,
    background_href=preview_background_href,
    df=st.session_state["objects_df"],
    keep_background=keep_background,
    background_opacity=background_opacity,
    show_edit_layer=show_edit_layer
)

components.html(
    preview_svg,
    height=min(height + 80, 900),
    scrolling=True
)

json_text = st.session_state["objects_df"].to_json(
    orient="records",
    force_ascii=False,
    indent=2
)

editable_vsdx_bytes = make_native_editable_vsdx(
    df=st.session_state["objects_df"],
    img_width_px=width,
    img_height_px=height,
    dpi=export_dpi
)

export_zip_bytes = make_export_zip(
    image=image,
    df=st.session_state["objects_df"],
    preview_svg=preview_svg,
    editable_vsdx_bytes=editable_vsdx_bytes,
    json_text=json_text
)

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        label="下载原生可编辑 VSDX",
        data=editable_vsdx_bytes,
        file_name="native_editable.vsdx",
        mime="application/vnd.ms-visio.drawing",
        use_container_width=True
    )

with download_col2:
    st.download_button(
        label="下载完整 ZIP 包",
        data=export_zip_bytes,
        file_name="image_to_visio_export_package.zip",
        mime="application/zip",
        use_container_width=True
    )

with download_col3:
    st.download_button(
        label="下载识别结果 JSON",
        data=json_text.encode("utf-8"),
        file_name="recognition_result.json",
        mime="application/json",
        use_container_width=True
    )

st.warning(
    "注意：native_editable.vsdx 是可编辑矢量对象版本，不包含原图底图。"
    "如果你把原图作为底图导入 Visio，它一定还是图片，不能逐个编辑。"
)

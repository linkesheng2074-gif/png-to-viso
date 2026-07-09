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


# =========================
# 页面配置
# =========================

st.set_page_config(
    page_title="图片转 Visio 可编辑文件 V3",
    layout="wide"
)

st.title("图片转 Visio 可编辑文件 V3")
st.caption(
    "上传图片 → 原图高保真底图 → 可选识别文字/线条/矩形 → 人工修正 → "
    "导出 Visio 兼容 ZIP。解压后用 Visio 打开 SVG。"
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
                df[col] = "#ff0000"
            elif col == "fill":
                df[col] = "none"
            elif col == "stroke_width":
                df[col] = 1
            elif col == "opacity":
                df[col] = 0.45
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
    stroke_color="#ff0000"
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
                "opacity": 0.45,
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
    stroke_color="#ff0000"
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
            "opacity": 0.45,
            "font_size": 12
        })

    return objects


# =========================
# OCR 文字识别
# =========================

def detect_text(
    image: Image.Image,
    min_conf=80,
    stroke_color="#0000ff"
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
                "opacity": 0.45,
                "font_size": font_size
            })

    except Exception as e:
        st.warning(f"OCR 文字识别失败，已跳过：{e}")

    return objects


# =========================
# 生成 SVG
# =========================

def make_svg(
    width,
    height,
    background_href,
    df,
    keep_background=True,
    background_opacity=1.0,
    show_edit_layer=False
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

            stroke = str(row.get("stroke", "#ff0000")).strip()
            fill = str(row.get("fill", "none")).strip()
            text = html.escape(str(row.get("text", "")))
            stroke_width = safe_float(row.get("stroke_width", 1), 1)
            opacity = safe_float(row.get("opacity", 0.45), 0.45)
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
                    f'font-family="Arial" '
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
# 生成 ZIP
# =========================

def make_visio_zip(
    image: Image.Image,
    svg_with_background: str,
    svg_overlay_only: str,
    json_text: str
):
    zip_buffer = BytesIO()

    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("background.png", img_bytes)
        zf.writestr("visio_compatible.svg", svg_with_background.encode("utf-8"))
        zf.writestr("overlay_only.svg", svg_overlay_only.encode("utf-8"))
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

ocr_conf = st.sidebar.slider(
    "OCR 最低置信度",
    min_value=0,
    max_value=100,
    value=80,
    step=5
)

overlay_color = st.sidebar.color_picker(
    "识别图层颜色",
    value="#ff0000"
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
        "建议：复杂封装图、规格书截图，先关闭 OCR，只识别线条/矩形；"
        "需要文字时再单独开启 OCR。"
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

        if enable_text:
            try:
                texts = detect_text(
                    image,
                    min_conf=ocr_conf,
                    stroke_color="#0000ff"
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
        "导出时保留原图作为底图",
        value=True
    )

with preview_col2:
    background_opacity = st.slider(
        "底图不透明度",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05
    )

with preview_col3:
    show_edit_layer = st.checkbox(
        "显示识别图层",
        value=False
    )


# 页面预览：可以使用 base64，因为这是给网页预览用，不给 Visio 导入
preview_background_href = f"data:image/png;base64,{image_to_base64(image)}"

preview_svg = make_svg(
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


# Visio 导出：不要使用 base64，使用相对路径 background.png
svg_with_background = make_svg(
    width=width,
    height=height,
    background_href="background.png",
    df=st.session_state["objects_df"],
    keep_background=keep_background,
    background_opacity=background_opacity,
    show_edit_layer=show_edit_layer
)

# 纯识别图层版本，兼容性更高，但没有底图
svg_overlay_only = make_svg(
    width=width,
    height=height,
    background_href="",
    df=st.session_state["objects_df"],
    keep_background=False,
    background_opacity=1.0,
    show_edit_layer=True
)

json_text = st.session_state["objects_df"].to_json(
    orient="records",
    force_ascii=False,
    indent=2
)

zip_bytes = make_visio_zip(
    image=image,
    svg_with_background=svg_with_background,
    svg_overlay_only=svg_overlay_only,
    json_text=json_text
)

download_col1, download_col2 = st.columns(2)

with download_col1:
    st.download_button(
        label="下载 Visio 兼容 ZIP",
        data=zip_bytes,
        file_name="visio_compatible_package.zip",
        mime="application/zip",
        use_container_width=True
    )

with download_col2:
    st.download_button(
        label="下载识别结果 JSON",
        data=json_text.encode("utf-8"),
        file_name="recognition_result.json",
        mime="application/json",
        use_container_width=True
    )


st.info(
    "使用方法：下载 ZIP 后先解压，保持 background.png 和 visio_compatible.svg 在同一个文件夹，"
    "然后用 Visio 打开或导入 visio_compatible.svg。"
)

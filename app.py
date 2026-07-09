import os
import html
import hashlib
import tempfile
import zipfile
from io import BytesIO

import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract

# =========================
# 可选：Aspose.Diagram
# 用来导出真正的 .vsdx
# =========================
ASPOSE_AVAILABLE = True
ASPOSE_IMPORT_ERROR = ""

try:
    import aspose.diagram as ad
    from aspose.diagram import Diagram, SaveFileFormat, License
except Exception as e:
    ASPOSE_AVAILABLE = False
    ASPOSE_IMPORT_ERROR = str(e)


# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="图片转 Visio（100%视觉还原版）",
    layout="wide"
)

st.title("图片转 Visio（100%视觉还原版）")
st.caption(
    "目标：导出与原图 100% 视觉一致的 VSDX。"
    "实现方式：将原图作为图片嵌入到 VSDX 页面中。"
)

st.warning(
    "重要说明：\n\n"
    "1. 这版是【100%视觉还原】模式，不是自动重绘模式；\n"
    "2. 导出的 VSDX 会和原图看起来完全一致；\n"
    "3. 但里面的主体内容本质上仍然是图片对象，不会自动变成每根线/每个字都可编辑的 Visio 图元；\n"
    "4. 如果你既要“100%一致”，又要“所有元素自动可编辑”，纯自动化目前做不到，必须走人工半自动校正。"
)

# =========================
# 基础列定义
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
# 工具函数
# =========================
def safe_float(value, default=0.0):
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
    return buffer.getvalue().hex()


def image_to_base64_standard(img: Image.Image) -> str:
    import base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def uploaded_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def reset_recognition():
    st.session_state.pop("objects_df", None)


def px_to_in(px, dpi=96):
    return float(px) / float(dpi)


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
# OCR 识别
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
# SVG 预览（仅网页预览调试）
# =========================
def make_preview_svg(
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
                    f'font-family="Arial, Microsoft YaHei" '
                    f'fill="{stroke}" opacity="{opacity}">{text}</text>'
                )

    svg.append("</svg>")
    return "\n".join(svg)


# =========================
# Aspose License（可选）
# =========================
def try_apply_aspose_license(license_file):
    if not ASPOSE_AVAILABLE:
        return False, f"Aspose 未安装：{ASPOSE_IMPORT_ERROR}"

    if license_file is None:
        return False, "未提供 license 文件"

    try:
        lic_bytes = license_file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".lic") as tmp:
            tmp.write(lic_bytes)
            tmp_path = tmp.name

        lic = License()
        lic.set_license(tmp_path)

        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return True, "Aspose license 已加载"
    except Exception as e:
        return False, f"License 加载失败：{e}"


# =========================
# 尝试设置页面大小
# 不同版本 Aspose 属性路径可能略有差异，所以做容错
# =========================
def try_set_page_size(page, width_in, height_in):
    ok = False

    # 尝试 page.page_sheet.page_props.page_width / page_height
    try:
        if hasattr(page, "page_sheet"):
            ps = page.page_sheet
            if hasattr(ps, "page_props"):
                props = ps.page_props

                if hasattr(props, "page_width") and hasattr(props.page_width, "value"):
                    props.page_width.value = width_in
                    ok = True

                if hasattr(props, "page_height") and hasattr(props.page_height, "value"):
                    props.page_height.value = height_in
                    ok = True

                if hasattr(props, "drawing_size_type") and hasattr(props.drawing_size_type, "value"):
                    props.drawing_size_type.value = 0
    except Exception:
        pass

    return ok


# =========================
# 导出：100%视觉还原 VSDX
# 原图作为图片嵌入 VSDX
# =========================
def export_exact_visual_vsdx(image_bytes: bytes, width_px: int, height_px: int, dpi: int = 96):
    if not ASPOSE_AVAILABLE:
        raise RuntimeError(f"Aspose.Diagram 不可用：{ASPOSE_IMPORT_ERROR}")

    width_in = px_to_in(width_px, dpi)
    height_in = px_to_in(height_px, dpi)

    diagram = Diagram()
    page = diagram.pages[0]

    # 尝试设置页面大小
    try_set_page_size(page, width_in, height_in)

    # 关键：原图作为图片放进页面，保证视觉 100% 一致
    img_stream = BytesIO(image_bytes)
    page.add_shape(
        width_in / 2,
        height_in / 2,
        width_in,
        height_in,
        img_stream
    )

    # 保存到临时文件，再读回 bytes
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".vsdx") as tmp:
            tmp_path = tmp.name

        diagram.save(tmp_path, SaveFileFormat.VSDX)

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# =========================
# 导出 ZIP 调试包
# =========================
def make_debug_zip(
    original_image_bytes: bytes,
    preview_svg: str,
    json_text: str,
    vsdx_bytes: bytes | None
):
    buffer = BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("original_image.png", original_image_bytes)
        zf.writestr("preview_debug.svg", preview_svg.encode("utf-8"))
        zf.writestr("recognition_result.json", json_text.encode("utf-8"))

        if vsdx_bytes is not None:
            zf.writestr("exact_visual.vsdx", vsdx_bytes)

    buffer.seek(0)
    return buffer.getvalue()


# =========================
# 侧边栏
# =========================
st.sidebar.header("导出模式")
st.sidebar.success("当前版本：100%视觉还原优先")

if ASPOSE_AVAILABLE:
    st.sidebar.info("Aspose.Diagram：已可用")
else:
    st.sidebar.error(f"Aspose.Diagram 不可用：{ASPOSE_IMPORT_ERROR}")

st.sidebar.divider()
st.sidebar.header("Aspose License（可选）")

license_file = st.sidebar.file_uploader(
    "上传 Aspose License 文件（可选）",
    type=["lic"]
)

if license_file is not None:
    ok, msg = try_apply_aspose_license(license_file)
    if ok:
        st.sidebar.success(msg)
    else:
        st.sidebar.warning(msg)

st.sidebar.divider()
st.sidebar.header("识别调试参数（仅预览用）")

enable_enhance = st.sidebar.checkbox("识别前增强图片", value=True)
enable_rect = st.sidebar.checkbox("识别矩形", value=True)
enable_line = st.sidebar.checkbox("识别线条", value=True)
enable_text = st.sidebar.checkbox("识别文字 OCR", value=False)

min_area = st.sidebar.slider("矩形最小面积", 100, 5000, 800, 100)
min_line_length = st.sidebar.slider("线条最小长度", 10, 300, 60, 10)
max_line_gap = st.sidebar.slider("线条最大断点连接距离", 1, 30, 8, 1)
only_hv = st.sidebar.checkbox("只识别水平/垂直线", value=True)
ocr_conf = st.sidebar.slider("OCR 最低置信度", 0, 100, 80, 5)
overlay_color = st.sidebar.color_picker("识别图层颜色", "#ff0000")

st.sidebar.divider()
export_dpi = st.sidebar.number_input(
    "导出 VSDX DPI",
    min_value=72,
    max_value=300,
    value=96,
    step=1
)


# =========================
# 上传图片
# =========================
uploaded = st.file_uploader(
    "上传图片",
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
# 页面主布局
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("原始图片")
    st.image(image, use_container_width=True)
    st.write(f"尺寸：{width} × {height} px")

with col2:
    st.subheader("功能说明")
    st.markdown(
        """
        **这版的重点：**

        - 点击 **“导出 100%视觉还原 VSDX”**  
          得到的 `.vsdx` 会和原图看起来完全一样。

        - 下方识别结果只是给你做调试参考，**不参与 100%还原导出**。

        - 如果你想要“自动识别出来的矩形、线条、文字”，可以在网页里看调试叠加层。
        """
    )

    run_recognition = st.button(
        "开始识别（仅预览调试）",
        type="secondary",
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
# 执行识别（仅用于调试预览）
# =========================
if run_recognition or "objects_df" not in st.session_state:
    objects = []

    with st.spinner("正在识别（仅用于调试预览）..."):
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
                st.warning(f"矩形识别失败：{e}")

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
                st.warning(f"线条识别失败：{e}")

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
                st.warning(f"文字识别失败：{e}")

    st.session_state["objects_df"] = ensure_df(pd.DataFrame(objects))


# =========================
# 人工修正表（仅调试）
# =========================
st.divider()
st.subheader("识别结果（仅调试）")

objects_df = ensure_df(
    st.session_state.get("objects_df", pd.DataFrame(columns=DEFAULT_COLUMNS))
)

edited_df = st.data_editor(
    objects_df,
    num_rows="dynamic",
    use_container_width=True,
    height=320,
    column_config={
        "type": st.column_config.SelectboxColumn(
            "type",
            options=["rect", "line", "text"],
            required=True
        ),
        "x": st.column_config.NumberColumn("x", step=1),
        "y": st.column_config.NumberColumn("y", step=1),
        "w": st.column_config.NumberColumn("w", step=1),
        "h": st.column_config.NumberColumn("h", step=1),
        "text": st.column_config.TextColumn("text"),
        "stroke": st.column_config.TextColumn("stroke"),
        "fill": st.column_config.TextColumn("fill"),
        "stroke_width": st.column_config.NumberColumn("stroke_width", min_value=0.1, step=0.5),
        "opacity": st.column_config.NumberColumn("opacity", min_value=0.0, max_value=1.0, step=0.05),
        "font_size": st.column_config.NumberColumn("font_size", min_value=1, step=1),
    },
    key="object_editor"
)

st.session_state["objects_df"] = ensure_df(edited_df)


# =========================
# 预览区域
# =========================
st.divider()
st.subheader("预览")

preview_col1, preview_col2, preview_col3 = st.columns(3)

with preview_col1:
    keep_background = st.checkbox("预览时显示原图", value=True)

with preview_col2:
    background_opacity = st.slider(
        "原图不透明度",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.05
    )

with preview_col3:
    show_edit_layer = st.checkbox(
        "预览时叠加识别图层（调试用）",
        value=False
    )

preview_background_href = f"data:image/png;base64,{image_to_base64_standard(image)}"

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


# =========================
# 导出区
# =========================
st.divider()
st.subheader("导出")

exact_vsdx_bytes = None
export_error = None

if ASPOSE_AVAILABLE:
    try:
        exact_vsdx_bytes = export_exact_visual_vsdx(
            image_bytes=file_bytes,
            width_px=width,
            height_px=height,
            dpi=export_dpi
        )
    except Exception as e:
        export_error = str(e)
else:
    export_error = f"Aspose.Diagram 不可用：{ASPOSE_IMPORT_ERROR}"

json_text = st.session_state["objects_df"].to_json(
    orient="records",
    force_ascii=False,
    indent=2
)

debug_zip = make_debug_zip(
    original_image_bytes=file_bytes,
    preview_svg=preview_svg,
    json_text=json_text,
    vsdx_bytes=exact_vsdx_bytes
)

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        label="下载 100%视觉还原 VSDX（推荐）",
        data=exact_vsdx_bytes if exact_vsdx_bytes is not None else b"",
        file_name="exact_visual.vsdx",
        mime="application/vnd.ms-visio.drawing",
        use_container_width=True,
        disabled=(exact_vsdx_bytes is None)
    )

with download_col2:
    st.download_button(
        label="下载调试包 ZIP",
        data=debug_zip,
        file_name="debug_package.zip",
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

if export_error:
    st.error(f"VSDX 导出失败：{export_error}")

st.info(
    "现在这版是你要的“100%还原原图”版本：\n\n"
    "- 导出的 exact_visual.vsdx 会和原图看起来一致；\n"
    "- 但它本质上是图片放进 Visio 页面中；\n"
    "- 如果你下一步要做“可编辑重绘版”，建议改成：图片做底图 + 人工半自动描图。"
)

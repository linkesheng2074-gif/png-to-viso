import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import base64
from io import BytesIO
import html

st.set_page_config(page_title="图片转 Visio 可编辑文件 V3", layout="wide")

st.title("图片转 Visio 可编辑文件 V3")
st.caption("上传图片 → 自动识别 → 人工修正 → 导出 SVG，可用 Visio 打开/导入后继续编辑。")


# =========================
# 工具函数
# =========================

def image_to_base64(img: Image.Image):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def detect_shapes(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objects = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < 300:
            continue

        if w > 15 and h > 15:
            objects.append({
                "type": "rect",
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "text": "",
                "stroke": "#000000",
                "fill": "none",
                "stroke_width": 1
            })

    return objects


def detect_lines(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=40,
        maxLineGap=8
    )

    objects = []

    if lines is not None:
        for line in lines:
            pts = np.array(line).reshape(-1)

            if len(pts) != 4:
                continue

            x1, y1, x2, y2 = pts

            objects.append({
                "type": "line",
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "text": "",
                "stroke": "#000000",
                "fill": "none",
                "stroke_width": 1
            })

    return objects


def detect_text(image: Image.Image):
    objects = []

    try:
        data = pytesseract.image_to_data(
            image,
            lang="chi_sim+eng",
            output_type=pytesseract.Output.DATAFRAME
        )

        data = data.dropna()

        for _, row in data.iterrows():
            text = str(row.get("text", "")).strip()

            if not text:
                continue

            x = int(row["left"])
            y = int(row["top"])
            w = int(row["width"])
            h = int(row["height"])

            if w < 5 or h < 5:
                continue

            objects.append({
                "type": "text",
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "text": text,
                "stroke": "#000000",
                "fill": "none",
                "stroke_width": 1
            })

    except Exception as e:
        st.warning(f"OCR 文字识别失败，已跳过：{e}")

    return objects


def make_svg(width, height, image_b64, df, keep_background=True):
    svg = []

    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )

    if keep_background:
        svg.append(
            f'<image href="data:image/png;base64,{image_b64}" '
            f'x="0" y="0" width="{width}" height="{height}" opacity="0.35"/>'
        )

    for _, row in df.iterrows():
        obj_type = str(row.get("type", "")).strip()

        try:
            x = int(float(row.get("x", 0)))
            y = int(float(row.get("y", 0)))
            w = int(float(row.get("w", 0)))
            h = int(float(row.get("h", 0)))
            stroke_width = int(float(row.get("stroke_width", 1)))
        except:
            continue

        stroke = str(row.get("stroke", "#000000"))
        fill = str(row.get("fill", "none"))
        text = html.escape(str(row.get("text", "")))

        if obj_type == "rect":
            svg.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            )

        elif obj_type == "line":
            svg.append(
                f'<line x1="{x}" y1="{y}" x2="{x + w}" y2="{y + h}" '
                f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
            )

        elif obj_type == "text":
            font_size = max(8, int(abs(h) * 1.2))
            svg.append(
                f'<text x="{x}" y="{y + h}" '
                f'font-size="{font_size}" '
                f'font-family="Arial, Microsoft YaHei" '
                f'fill="{stroke}">{text}</text>'
            )

    svg.append("</svg>")

    return "\n".join(svg)


def reset_session():
    if "objects_df" in st.session_state:
        del st.session_state["objects_df"]


# =========================
# 页面主体
# =========================

uploaded = st.file_uploader(
    "上传需要还原的图片",
    type=["png", "jpg", "jpeg", "bmp"],
    on_change=reset_session
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_np = np.array(image)
    width, height = image.size

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("原始图片")
        st.image(image, use_container_width=True)
        st.write(f"图片尺寸：{width} × {height} px")

    if "objects_df" not in st.session_state:
        with st.spinner("正在识别图片元素..."):
            objects = []

            try:
                objects += detect_shapes(image_np)
            except Exception as e:
                st.warning(f"矩形识别失败，已跳过：{e}")

            try:
                objects += detect_lines(image_np)
            except Exception as e:
                st.warning(f"线条识别失败，已跳过：{e}")

            try:
                objects += detect_text(image)
            except Exception as e:
                st.warning(f"文字识别失败，已跳过：{e}")

            if len(objects) == 0:
                objects = [{
                    "type": "rect",
                    "x": 10,
                    "y": 10,
                    "w": 100,
                    "h": 50,
                    "text": "",
                    "stroke": "#000000",
                    "fill": "none",
                    "stroke_width": 1
                }]

            st.session_state.objects_df = pd.DataFrame(objects)

    with col2:
        st.subheader("识别结果 / 人工修正")
        st.write("可修改 x、y、w、h、文字、颜色、线宽；也可以新增或删除行。")

        edited_df = st.data_editor(
            st.session_state.objects_df,
            num_rows="dynamic",
            use_container_width=True,
            key="editor"
        )

        st.session_state.objects_df = edited_df

    st.divider()

    keep_background = st.checkbox(
        "导出时保留原图作为半透明底图",
        value=True
    )

    image_b64 = image_to_base64(image)

    svg_text = make_svg(
        width=width,
        height=height,
        image_b64=image_b64,
        df=st.session_state.objects_df,
        keep_background=keep_background
    )

    st.subheader("SVG 预览")
    st.components.v1.html(
        svg_text,
        height=min(height + 100, 800),
        scrolling=True
    )

    st.download_button(
        label="下载 SVG 文件，可用 Visio 打开/导入",
        data=svg_text.encode("utf-8"),
        file_name="image_to_visio_editable.svg",
        mime="image/svg+xml"
    )

    json_text = st.session_state.objects_df.to_json(
        orient="records",
        force_ascii=False,
        indent=2
    )

    st.download_button(
        label="下载识别结果 JSON",
        data=json_text.encode("utf-8"),
        file_name="recognition_result.json",
        mime="application/json"
    )

else:
    st.info("请先上传图片。")

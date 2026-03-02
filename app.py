import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from pdf2image import convert_from_bytes

# ==========================================
# 高精度・画像処理ユーティリティ群
# ==========================================

def deskew(pil_img):
    """画像のわずかな傾きを補正"""
    img_array = np.array(pil_img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0: return pil_img
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    (h, w) = img_array.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return Image.fromarray(cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))

def detect_staff_lines_precise(pil_img):
    """水平線抽出に特化した五線検出"""
    img_array = np.array(pil_img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    height, width = gray.shape
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 水平な長い直線だけを抽出
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 20, 1))
    clean_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 10, 1))
    clean_lines = cv2.dilate(clean_lines, bridge_kernel, iterations=1)

    contours, _ = cv2.findContours(clean_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    staff_y_coords = sorted([int(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]//2) for c in contours if cv2.boundingRect(c)[2] > width // 5])

    # 近接線のマージ
    merged_y = []
    for y in staff_y_coords:
        if not merged_y or y - merged_y[-1] > 5: merged_y.append(y)
        else: merged_y[-1] = (merged_y[-1] + y) // 2

    # 五線間隔の推定
    staff_space = 10
    if len(merged_y) >= 2:
        diffs = np.diff(merged_y)
        valid_diffs = diffs[(diffs > 3) & (diffs < height // 20)]
        if len(valid_diffs) > 0: staff_space = np.median(valid_diffs)

    return merged_y, gray, staff_space

def non_max_suppression_fast(boxes, overlapThresh):
    """重複した検出枠を統合（感度を上げた際の重複対策）"""
    if len(boxes) == 0: return []
    if boxes.dtype.kind == "i": boxes = boxes.astype("float")

    pick = []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]
        # 閾値を厳しく（0.3）して、少しでも重なったら統合する
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > 0.3)[0])))

    return boxes[pick].astype("int")

def detect_note_heads_precise(gray_img, staff_space, user_threshold):
    """音符の頭を検出"""
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_size = max(2, int(staff_space * 0.4)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    nw, nh, pad = int(staff_space * 1.2), int(staff_space * 0.9), 2
    template = np.zeros((nh + pad*2, nw + pad*2), dtype=np.uint8)
    cv2.ellipse(template, (nw//2+pad, nh//2+pad), (nw//2, nh//2), -20, 0, 360, 255, -1)
    
    res = cv2.matchTemplate(clean_thresh, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= user_threshold)
    rects = [[int(pt[0]+pad), int(pt[1]+pad), int(pt[0]+pad+nw), int(pt[1]+pad+nh)] for pt in zip(*loc[::-1])]
    
    if not rects: return []
    # 強力な重複除去を適用
    picked_boxes = non_max_suppression_fast(np.array(rects), overlapThresh=0.3)
    
    final_boxes = []
    for (x1, y1, x2, y2) in picked_boxes:
        roi = thresh[y1:y2, x1:x2]
        if roi.size > 0 and 0.35 < (np.count_nonzero(roi) / roi.size) < 0.95:
            final_boxes.append([x1, y1, x2, y2])
    return final_boxes

# ==========================================
# 音階判定ロジック（調号・色対応）
# ==========================================

def get_pitch_info(note_y, staff_lines, clef, flats_count):
    """音名と、臨時記号（♭）が付くかどうかの情報を返す"""
    top_line = staff_lines[0]
    bottom_line = staff_lines[4]
    step_size = (bottom_line - top_line) / 8.0
    steps_from_top = round((note_y - top_line) / step_size)

    # 音名マッピング（0=第5線基準）
    if clef == "treble":
        mapping = {
            -4:"レ", -3:"ド", -2:"シ", -1:"ラ", 
            0:"ファ", 1:"ミ", 2:"レ", 3:"ド", 4:"シ", 5:"ラ", 6:"ソ", 7:"ファ", 8:"ミ",
            9:"レ", 10:"ド", 11:"シ", 12:"ラ"
        }
    else: # bass
        mapping = {
            -4:"シ", -3:"ラ", -2:"ソ", -1:"ファ",
            0:"ラ", 1:"ソ", 2:"ファ", 3:"ミ", 4:"レ", 5:"ド", 6: "シ", 7: "ラ", 8: "ソ",
            9:"ファ", 10:"ミ", 11:"レ", 12:"ド"
        }

    base_name = mapping.get(steps_from_top, "")
    if not base_name: return "", False

    # 調号（♭）の適用判定
    flat_order = ["シ", "ミ", "ラ", "レ", "ソ", "ド", "ファ"]
    is_flat = base_name in flat_order[:flats_count]
    
    return base_name, is_flat

# ==========================================
# 解析マスター関数
# ==========================================

def analyze_score_final_v2(pil_img, threshold, flats_count):
    img = deskew(pil_img)
    y_coords, gray, space = detect_staff_lines_precise(img)
    if len(y_coords) < 5: return img

    # 五線の段落分け
    staves = []
    for i in range(len(y_coords) - 4):
        lines = y_coords[i:i+5]
        if np.all(np.abs(np.diff(lines) - space) < space * 0.5):
            if not staves or lines[0] > staves[-1][4] + space: staves.append(lines)

    notes = detect_note_heads_precise(gray, space, threshold)
    result = img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    try: font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(13, int(space)))
    except: font = ImageFont.load_default()

    for box in notes:
        yc = (box[1] + box[3]) // 2
        # 最も近い五線グループを特定
        s_idx = np.argmin([abs(np.mean(s) - yc) for s in staves])
        clef = "treble" if s_idx % 2 == 0 else "bass"
        
        # 音名とフラット情報を取得
        p_name, is_flat = get_pitch_info(yc, staves[s_idx], clef, flats_count)
        
        if p_name:
            draw.rectangle(box, outline=(0, 255, 0), width=2)
            
            # 【新仕様】通常は赤、フラット（is_flat=True）なら青
            text_color = (0, 0, 255) if is_flat else (255, 0, 0)
            
            # 文字（「♭」なし）を描画
            draw.text((box[0], box[1] - int(space * 1.3)), p_name, font=font, fill=text_color)

    return result

# ==========================================
# Streamlit UI
# ==========================================

st.set_page_config(page_title="ドレミ付与ツール V2", layout="centered")
st.title("🎼 楽譜ドレミ付与ツール V2")
st.write("感度を上げても枠が重複しにくくなり、♭は文字の色（青）で判別できるようになりました。")

st.sidebar.header("⚙️ 設定")
flats = st.sidebar.selectbox("調号（♭）の数", range(8), index=4) # 小犬のワルツ用
sens = st.sidebar.slider("検出感度（低いほど多く検出）", 0.4, 0.95, 0.65, 0.01)

up = st.file_uploader("PDF形式の楽譜をアップロードしてください", type="pdf")

if up:
    imgs = convert_from_bytes(up.read())
    with st.spinner('解析中...'):
        processed = [analyze_score_final_v2(im, sens, flats) for im in imgs]
        for i, im in enumerate(processed):
            st.image(im, caption=f"{i+1}ページ目", use_column_width=True)

    # PDFにまとめてダウンロード
    pdf_buf = io.BytesIO()
    processed[0].save(pdf_buf, format='PDF', save_all=True, append_images=processed[1:])
    st.download_button(label="完成版PDFをダウンロード", data=pdf_buf.getvalue(), file_name="doremi_score_v2.pdf", mime="application/pdf")

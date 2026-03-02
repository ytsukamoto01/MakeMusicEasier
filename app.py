import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from pdf2image import convert_from_bytes

# --- 画像処理・検出ロジック ---

def deskew(pil_img):
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
    img_array = np.array(pil_img)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) if len(img_array.shape) == 3 else img_array
    height, width = gray.shape
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 水平線のみを物理的に抽出
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 20, 1))
    clean_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 10, 1))
    clean_lines = cv2.dilate(clean_lines, bridge_kernel, iterations=1)

    contours, _ = cv2.findContours(clean_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    staff_y_coords = sorted([int(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3]//2) for c in contours if cv2.boundingRect(c)[2] > width // 5])

    merged_y = []
    for y in staff_y_coords:
        if not merged_y or y - merged_y[-1] > 5: merged_y.append(y)
        else: merged_y[-1] = (merged_y[-1] + y) // 2

    staff_space = 10
    if len(merged_y) >= 2:
        diffs = np.diff(merged_y)
        valid_diffs = diffs[(diffs > 3) & (diffs < height // 20)]
        if len(valid_diffs) > 0: staff_space = np.median(valid_diffs)

    return merged_y, gray, staff_space

def detect_note_heads(gray_img, staff_space, user_threshold):
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
    rects.sort(key=lambda x: (x[1], x[0]))
    final = []
    for r in rects:
        if not final or abs(r[0]-final[-1][0]) > nw//2 or abs(r[1]-final[-1][1]) > nh//2: final.append(r)
    return final

# --- 修正版：音階判定ロジック ---

def get_accurate_pitch(note_y, staff_lines, clef, flats_count):
    # 第1線(下)から第5線(上)までの5本
    # staff_lines[0] が一番上、staff_lines[4] が一番下
    top_line = staff_lines[0]
    bottom_line = staff_lines[4]
    
    # 1ステップ（線と間の距離）を算出
    step_size = (bottom_line - top_line) / 8.0
    
    # 第5線（一番上の線）からの距離でステップ数を計算
    # 0 = 第5線, 1 = 第4間, 2 = 第4線 ...
    steps_from_top = round((note_y - top_line) / step_size)

    # 音名マッピング（0=第5線基準）
    if clef == "treble":
        # 0:ファ, 1:ミ, 2:レ, 3:ド, 4:シ, 5:ラ, 6:ソ, 7:ファ, 8:ミ
        # 今回のご指摘：第2間（下から2番目の隙間）は steps_from_top = 5 になるはず
        mapping = {
            -4:"レ", -3:"ド", -2:"シ", -1:"ラ", 
            0:"ファ", 1:"ミ", 2:"レ", 3:"ド", 4:"シ", 5:"ラ", 6:"ソ", 7:"ファ", 8:"ミ",
            9:"レ", 10:"ド", 11:"シ", 12:"ラ"
        }
    else: # bass
        # 0:ラ, 1:ソ, 2:ファ, 3:ミ, 4:レ, 5:ド, 6:シ, 7:ラ, 8:ソ
        mapping = {
            -4:"シ", -3:"ラ", -2:"ソ", -1:"ファ",
            0:"ラ", 1:"ソ", 2:"ファ", 3:"ミ", 4:"レ", 5:"ド", 6:"シ", 7:"ラ", 8:"ソ",
            9:"ファ", 10:"ミ", 11:"レ", 12:"ド"
        }

    base_name = mapping.get(steps_from_top, "")
    if not base_name: return ""

    # 調号（♭）の適用
    flat_order = ["シ", "ミ", "ラ", "レ", "ソ", "ド", "ファ"]
    if base_name in flat_order[:flats_count]:
        return base_name + "♭"
    return base_name

# --- メイン処理 ---

def analyze_score_final(pil_img, threshold, flats_count):
    img = deskew(pil_img)
    y_coords, gray, space = detect_staff_lines_precise(img)
    if len(y_coords) < 5: return img

    # 五線の段落分け
    staves = []
    for i in range(len(y_coords) - 4):
        lines = y_coords[i:i+5]
        if np.all(np.abs(np.diff(lines) - space) < space * 0.5):
            if not staves or lines[0] > staves[-1][4] + space: staves.append(lines)

    notes = detect_note_heads(gray, space, threshold)
    result = img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    try: font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(13, int(space)))
    except: font = ImageFont.load_default()

    for box in notes:
        yc = (box[1] + box[3]) // 2
        # 最も近い五線グループを特定
        s_idx = np.argmin([abs(np.mean(s) - yc) for s in staves])
        clef = "treble" if s_idx % 2 == 0 else "bass"
        
        p_name = get_accurate_pitch(yc, staves[s_idx], clef, flats_count)
        if p_name:
            draw.rectangle(box, outline=(0, 255, 0), width=2)
            # 文字位置を少し調整
            draw.text((box[0], box[1] - int(space * 1.3)), p_name, font=font, fill=(255, 0, 0))

    return result

# --- UI ---
st.title("🎼 ドレミ付与ツール [本番・最終版]")
flats = st.sidebar.selectbox("♭の数", range(8), index=4) # 小犬のワルツ用
sens = st.sidebar.slider("検出感度", 0.4, 0.95, 0.65)
up = st.file_uploader("PDFをアップロード", type="pdf")

if up:
    imgs = convert_from_bytes(up.read())
    with st.spinner('解析中...'):
        processed = [analyze_score_final(im, sens, flats) for im in imgs]
        for im in processed: st.image(im, use_column_width=True)

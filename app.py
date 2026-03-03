import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from pdf2image import convert_from_bytes

# ==========================================
# 1. 楽譜解析エンジン（V8：オープニング処理による五線・符幹の完全分離版）
# ==========================================

def detect_staff_groups_v8(pil_img):
    img_array = np.array(pil_img.convert('L'))
    _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = np.sum(thresh, axis=1)
    
    peaks = []
    thresh_val = np.max(projection) * 0.5
    for i in range(1, len(projection) - 1):
        if projection[i] > thresh_val and projection[i] >= projection[i-1] and projection[i] >= projection[i+1]:
            peaks.append(i)

    merged = []
    if peaks:
        curr = peaks[0]
        for i in range(1, len(peaks)):
            if peaks[i] - curr < 3:
                curr = (curr + peaks[i]) // 2
            else:
                merged.append(curr)
                curr = peaks[i]
        merged.append(curr)

    staves = []
    i = 0
    avg_spacing = 10
    while i <= len(merged) - 5:
        segment = merged[i:i+5]
        diffs = np.diff(segment)
        avg_spacing = np.mean(diffs)
        if np.all(np.abs(diffs - avg_spacing) < avg_spacing * 0.25):
            staves.append(sorted(segment, reverse=True))
            i += 5
        else:
            i += 1
    return staves, avg_spacing if staves else 10

def nms_v8_strict(boxes, scores, staff_space):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    pick = []
    
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        
        c_x = (boxes[i, 0] + boxes[i, 2]) / 2
        c_y = (boxes[i, 1] + boxes[i, 3]) / 2
        
        remaining = idxs[1:]
        if len(remaining) == 0:
            break
        
        o_x = (boxes[remaining, 0] + boxes[remaining, 2]) / 2
        o_y = (boxes[remaining, 1] + boxes[remaining, 3]) / 2
        
        dist = np.sqrt((c_x - o_x)**2 + (c_y - o_y)**2)
        
        duplicate = (dist < staff_space * 0.7)
        idxs = np.delete(remaining, np.where(duplicate)[0])
        
    return boxes[pick].astype("int")

def detect_note_heads_v8(gray_img, staff_space, threshold_val, staves):
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ==========================================
    # ★改良：オープニング処理で「細い線」をすべて溶かす
    # 五線（横線）や符幹（縦の棒）は細いため消滅し、
    # 符頭（黒くて丸い部分）だけが残ります。
    # ==========================================
    open_k_size = max(3, int(staff_space * 0.6))  # 音符より少し小さい円
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k_size, open_k_size))
    notes_only = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)

    # 溶けすぎた部分を少しだけ復元する
    close_k_size = max(3, int(staff_space * 0.3))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k_size, close_k_size))
    filled = cv2.morphologyEx(notes_only, cv2.MORPH_CLOSE, close_k)

    # テンプレートマッチング
    nw, nh = int(staff_space * 1.3), int(staff_space * 1.0)
    template = np.zeros((nh, nw), dtype=np.uint8)
    cv2.ellipse(
        template,
        (nw // 2, nh // 2),
        (nw // 2 - 1, nh // 2 - 1),
        -20, 0, 360, 255, -1
    )

    res = cv2.matchTemplate(filled, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold_val)

    staff_centers = [np.mean(s) for s in staves]
    raw_rects = []
    raw_scores = []

    for (x, y) in zip(*loc[::-1]):
        w, h = nw, nh
        score = res[y, x]

        x1, y1, x2, y2 = x, y, x + w, y + h
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # 1) 五線からの距離
        dist_to_nearest_staff = min(abs(cy - c) for c in staff_centers)
        if dist_to_nearest_staff > staff_space * 4.0:
            continue

        # 2) アスペクト比（ノイズが消えて形が綺麗に出るため、条件を少し広げる）
        aspect = w / float(h)
        if not (0.8 <= aspect <= 2.2):
            continue

        # 3) 面積チェック
        area = w * h
        if area < (staff_space**2) * 0.3:
            continue
        if area > (staff_space**2) * 3.5:
            continue

        # 4) 丸さチェック
        patch = filled[max(0, y1):y2, max(0, x1):x2]
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        if cnt_area == 0:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # 余計な線が消えて純粋な丸になるため、0.50前後でも通るようになる
        circularity = 4.0 * np.pi * cnt_area / (perimeter**2)
        if circularity < 0.40:  
            continue

        raw_rects.append([x1, y1, x2, y2])
        raw_scores.append(score)

    if not raw_rects:
        return []

    boxes = np.array(raw_rects)
    scores = np.array(raw_scores)
    return nms_v8_strict(boxes, scores, staff_space)

def get_pitch_name(note_y, staff, clef, flats_count):
    line1, line5 = staff[0], staff[4]
    step_size = abs(line1 - line5) / 8.0
    steps = int(round((line1 - note_y) / step_size))

    if clef == "treble":
        mapping = {
            -4: "ラ", -3: "シ", -2: "ド", -1: "レ",
             0: "ミ",  1: "ファ", 2: "ソ",  3: "ラ",
             4: "シ",  5: "ド",  6: "レ",  7: "ミ",
             8: "ファ", 9: "ソ", 10: "ラ", 11: "シ",
            12: "ド", 13: "レ", 14: "ミ", 15: "ファ",
            16: "ソ"
        }
    else:
        mapping = {
            -6: "ド", -5: "レ", -4: "ミ", -3: "ファ",
            -2: "ソ", -1: "ラ",  0: "シ",  1: "ド",
             2: "レ",  3: "ミ",  4: "ファ", 5: "ソ",
             6: "ラ",  7: "シ",  8: "ド",  9: "レ",
            10: "ミ", 11: "ファ", 12: "ソ", 13: "ラ",
            14: "シ"
        }

    name = mapping.get(steps, "")
    flat_order = ["シ", "ミ", "ラ", "レ", "ソ", "ド", "ファ"]
    return name, name in flat_order[:flats_count]

def process_page_v8(pil_img, internal_threshold, flats_count):
    staves, space = detect_staff_groups_v8(pil_img)
    if not staves:
        return pil_img

    img_gray = np.array(pil_img.convert('L'))

    notes = detect_note_heads_v8(img_gray, space, internal_threshold, staves)

    result = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(15, int(space)))
    except:
        font = ImageFont.load_default()

    for box in notes:
        b = box.tolist()
        yc = (b[1] + b[3]) // 2
        s_idx = np.argmin([abs(np.mean(s) - yc) for s in staves])
        clef = "treble" if s_idx % 2 == 0 else "bass"
        p_name, is_flat = get_pitch_name(yc, staves[s_idx], clef, flats_count)
        if p_name:
            draw.rectangle(b, outline=(0, 255, 0), width=2)
            color = (0, 0, 255) if is_flat else (255, 0, 0)
            draw.text((b[0], b[1] - int(space * 1.6)), p_name, font=font, fill=color)

    return result

# ==========================================
# 2. Streamlit UI
# ==========================================

st.set_page_config(page_title="ドレミ付与 V8", layout="centered")
st.title("🎼 ドレミ付与ツール V8 (スマートノイズ除去モード)")
st.write("「細い線」だけを溶かす処理により、音符の形を崩さずに記号や五線を綺麗に除外します。")

st.sidebar.header("⚙️ 設定")
flats = st.sidebar.selectbox("調号（♭）の数", range(8), index=4)

ui_sens = st.sidebar.slider("検出感度（1〜100：大きいほどたくさん検出）", min_value=1, max_value=100, value=50, step=1)

# ノイズが綺麗に消えるため、しきい値の感度幅をより広く（0.85〜0.45）取れるようにしました
internal_threshold = 0.85 - (ui_sens / 100.0) * 0.40

up = st.file_uploader("PDFをアップロード", type="pdf")
if up:
    imgs = convert_from_bytes(up.read())
    with st.spinner('解析中...'):
        for im in [process_page_v8(i, internal_threshold, flats) for i in imgs]:
            st.image(im, use_column_width=True)

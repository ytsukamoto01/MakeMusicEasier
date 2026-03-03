import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import math
from pdf2image import convert_from_bytes
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. 楽譜解析エンジン (V8 ロジック)
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
        remaining = idxs[1:]
        if len(remaining) == 0:
            break
        o_x = (boxes[remaining, 0] + boxes[remaining, 2]) / 2
        o_y = (boxes[remaining, 1] + boxes[remaining, 3]) / 2
        dist = np.sqrt((c_x - o_x)**2 + ((boxes[i, 1] + boxes[i, 3])/2 - o_y)**2)
        duplicate = (dist < staff_space * 0.7)
        idxs = np.delete(remaining, np.where(duplicate)[0])
    return boxes[pick].astype("int")

def detect_note_heads_v8(gray_img, staff_space, threshold_val, staves):
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    open_k_size = max(3, int(staff_space * 0.6))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k_size, open_k_size))
    notes_only = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
    close_k_size = max(3, int(staff_space * 0.3))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k_size, close_k_size))
    filled = cv2.morphologyEx(notes_only, cv2.MORPH_CLOSE, close_k)

    nw, nh = int(staff_space * 1.3), int(staff_space * 1.0)
    template = np.zeros((nh, nw), dtype=np.uint8)
    cv2.ellipse(template, (nw // 2, nh // 2), (nw // 2 - 1, nh // 2 - 1), -20, 0, 360, 255, -1)
    res = cv2.matchTemplate(filled, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold_val)

    staff_centers = [np.mean(s) for s in staves]
    raw_rects, raw_scores = [], []
    for (x, y) in zip(*loc[::-1]):
        w, h = nw, nh
        score = res[y, x]
        cx, cy = x + w//2, y + h//2
        dist_to_nearest_staff = min(abs(cy - c) for c in staff_centers)
        if dist_to_nearest_staff > staff_space * 4.0: continue
        aspect = w / float(h)
        if not (0.8 <= aspect <= 2.2): continue
        if not ((staff_space**2)*0.3 < w*h < (staff_space**2)*3.5): continue
        
        patch = filled[y:y+h, x:x+w]
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri > 0 and (4.0 * np.pi * area / (peri**2)) >= 0.40:
                raw_rects.append([x, y, x+w, y+h])
                raw_scores.append(score)
    
    return nms_v8_strict(np.array(raw_rects), np.array(raw_scores), staff_space) if raw_rects else []

def get_pitch_name(note_y, staff, clef, flats_count):
    line1, line5 = staff[0], staff[4]
    step_size = abs(line1 - line5) / 8.0
    steps = int(round((line1 - note_y) / step_size))
    if clef == "treble":
        mapping = {-4:"ラ",-3:"シ",-2:"ド",-1:"レ",0:"ミ",1:"ファ",2:"ソ",3:"ラ",4:"シ",5:"ド",6:"レ",7:"ミ",8:"ファ",9:"ソ",10:"ラ",11:"シ",12:"ド",13:"レ",14:"ミ",15:"ファ",16:"ソ"}
    else:
        mapping = {-6:"ド",-5:"レ",-4:"ミ",-3:"ファ",-2:"ソ",-1:"ラ",0:"シ",1:"ド",2:"レ",3:"ミ",4:"ファ",5:"ソ",6:"ラ",7:"シ",8:"ド",9:"レ",10:"ミ",11:"ファ",12:"ソ",13:"ラ",14:"シ"}
    name = mapping.get(steps, "")
    flat_order = ["シ", "ミ", "ラ", "レ", "ソ", "ド", "ファ"]
    return name, name in flat_order[:flats_count]

# ==========================================
# 2. 描画・キャッシュ処理
# ==========================================
def draw_all_notes(pil_img, auto_notes, custom_clicks, deleted_auto, staves, space, flats_count):
    result = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    
    font_size = max(15, int(space))
    try:
        font = ImageFont.truetype("NotoSansJP-Regular.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # ★追加：文字の重なりを防ぐための記録リスト
    drawn_text_rects = []

    def draw_label(x, yc, is_custom=False):
        s_idx = np.argmin([abs(np.mean(s) - yc) for s in staves])
        clef = "treble" if s_idx % 2 == 0 else "bass"
        p_name, is_flat = get_pitch_name(yc, staves[s_idx], clef, flats_count)
        
        if p_name:
            # 枠線の描画
            box_color = (255, 165, 0) if is_custom else (0, 255, 0)
            hw, hh = int(space * 0.6), int(space * 0.5)
            b = [x - hw, yc - hh, x + hw, yc + hh]
            draw.rectangle(b, outline=box_color, width=2)
            
            # 文字色
            color = (0, 0, 255) if is_flat else (255, 0, 0)
            
            # 初期描画位置
            text_x = b[0]
            text_y = b[1] - int(space * 1.6)
            
            # テキストの幅と高さを概算（重なり判定用）
            # 日本語フォントの場合、1文字の幅はフォントサイズとほぼ同じ
            text_w = len(p_name) * font_size
            text_h = font_size
            
            # ★追加：重なり（衝突）判定ループ
            while True:
                is_overlapping = False
                for (rx, ry, rw, rh) in drawn_text_rects:
                    # 矩形同士の重なりをチェック
                    if not (text_x + text_w < rx or text_x > rx + rw or text_y + text_h < ry or text_y > ry + rh):
                        is_overlapping = True
                        break
                
                if is_overlapping:
                    # 重なっている場合は右にズラす
                    text_x += int(font_size * 1.1)
                else:
                    # 重ならない場所が見つかったらループを抜ける
                    break

            # 決定した位置に文字を描画し、リストに記録
            draw.text((text_x, text_y), p_name, font=font, fill=color)
            drawn_text_rects.append((text_x, text_y, text_w, text_h))

    # 自動検出分の描画
    for box in auto_notes:
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        is_deleted = False
        for dx, dy in deleted_auto:
            if math.hypot(cx - dx, cy - dy) < 1.0:
                is_deleted = True
                break
        if not is_deleted:
            draw_label(cx, cy, False)

    # 手動追加分の描画
    for (cx, yc) in custom_clicks:
        draw_label(cx, yc, True)
        
    return result

@st.cache_data(show_spinner=False)
def process_pdf_and_detect(pdf_bytes, internal_threshold):
    imgs = convert_from_bytes(pdf_bytes)
    data = []
    for img in imgs:
        staves, space = detect_staff_groups_v8(img)
        notes = detect_note_heads_v8(np.array(img.convert('L')), space, internal_threshold, staves) if staves else []
        data.append({"image": img, "staves": staves, "space": space, "notes": notes})
    return data

# ==========================================
# 3. Streamlit UI
# ==========================================
st.set_page_config(page_title="ドレミ付与 V8", layout="wide") 
st.title("🎼 ドレミ付与ツール V8")
st.info("💡 **操作方法:** 検出漏れの場所をクリックすると追加されます。すでにある音符（枠線）の近くをクリックすると削除されます。")

if "custom_clicks" not in st.session_state:
    st.session_state.custom_clicks = {}
if "deleted_auto_notes" not in st.session_state:
    st.session_state.deleted_auto_notes = {}

st.sidebar.header("⚙️ 設定")
flats = st.sidebar.selectbox("調号（♭）の数", range(8), index=4)
ui_sens = st.sidebar.slider("検出感度", 1, 100, 50)
internal_threshold = 0.85 - (ui_sens / 100.0) * 0.40
disp_width = st.sidebar.slider("表示サイズ (px)", 400, 1500, 800)

up = st.file_uploader("PDFをアップロード", type="pdf")
if up:
    pages = process_pdf_and_detect(up.read(), internal_threshold)
    for i, page in enumerate(pages):
        st.write(f"### ページ {i + 1}")
        if page["staves"]:
            clicks = st.session_state.custom_clicks.get(i, [])
            deleted_auto = st.session_state.deleted_auto_notes.get(i, [])
            
            res_img = draw_all_notes(page["image"], page["notes"], clicks, deleted_auto, page["staves"], page["space"], flats)
            
            value = streamlit_image_coordinates(res_img, key=f"img_{i}", width=disp_width)
            
            last_click_key = f"last_click_{i}"
            if last_click_key not in st.session_state:
                st.session_state[last_click_key] = None
            
            if value and value != st.session_state[last_click_key]:
                st.session_state[last_click_key] = value
                
                clicked_x, clicked_y = value["x"], value["y"]
                scale = page["image"].width / disp_width
                real_x, real_y = clicked_x * scale, clicked_y * scale
                
                click_threshold = page["space"] * 1.5 
                action_taken = False

                for pt in clicks.copy():
                    if math.hypot(real_x - pt[0], real_y - pt[1]) < click_threshold:
                        clicks.remove(pt)
                        st.session_state.custom_clicks[i] = clicks
                        action_taken = True
                        break

                if not action_taken:
                    for box in page["notes"]:
                        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                        is_already_deleted = any(math.hypot(cx - dx, cy - dy) < 1.0 for dx, dy in deleted_auto)
                        if not is_already_deleted and math.hypot(real_x - cx, real_y - cy) < click_threshold:
                            if i not in st.session_state.deleted_auto_notes:
                                st.session_state.deleted_auto_notes[i] = []
                            st.session_state.deleted_auto_notes[i].append((cx, cy))
                            action_taken = True
                            break

                if not action_taken:
                    if i not in st.session_state.custom_clicks: 
                        st.session_state.custom_clicks[i] = []
                    st.session_state.custom_clicks[i].append((real_x, real_y))
                
                st.rerun() 
        else:
            st.image(page["image"], width=disp_width)

    if st.button("手動編集をすべてリセット"):
        st.session_state.custom_clicks = {}
        st.session_state.deleted_auto_notes = {}
        for key in list(st.session_state.keys()):
            if key.startswith("last_click_"):
                del st.session_state[key]
        st.rerun()

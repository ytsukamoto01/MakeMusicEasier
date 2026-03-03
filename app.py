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
    thresh_val = np.max(projection) * 0.3 
    for i in range(1, len(projection) - 1):
        if projection[i] > thresh_val and projection[i] >= projection[i-1] and projection[i] >= projection[i+1]:
            peaks.append(i)

    merged = []
    if peaks:
        curr = peaks[0]
        for i in range(1, len(peaks)):
            if peaks[i] - curr < 5: 
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
        if np.all(np.abs(diffs - avg_spacing) < avg_spacing * 0.45):
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

def get_pitch_name(note_y, staff, clef):
    line1, line5 = staff[0], staff[4]
    step_size = abs(line1 - line5) / 8.0
    steps = int(round((line1 - note_y) / step_size))
    if clef == "treble":
        mapping = {-4:"ラ",-3:"シ",-2:"ド",-1:"レ",0:"ミ",1:"ファ",2:"ソ",3:"ラ",4:"シ",5:"ド",6:"レ",7:"ミ",8:"ファ",9:"ソ",10:"ラ",11:"シ",12:"ド",13:"レ",14:"ミ",15:"ファ",16:"ソ"}
    else:
        mapping = {-6:"ド",-5:"レ",-4:"ミ",-3:"ファ",-2:"ソ",-1:"ラ",0:"シ",1:"ド",2:"レ",3:"ミ",4:"ファ",5:"ソ",6:"ラ",7:"シ",8:"ド",9:"レ",10:"ミ",11:"ファ",12:"ソ",13:"ラ",14:"シ"}
    return mapping.get(steps, "")

# ==========================================
# 2. 描画・キャッシュ処理 
# ==========================================
def draw_all_notes(pil_img, auto_notes, custom_clicks, deleted_auto, staves, space, custom_labels, hide_boxes=False, selected_pos=None):
    result = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    
    font_size = max(15, int(space))
    try:
        font = ImageFont.truetype("NotoSansJP-Regular.ttf", font_size)
    except:
        font = ImageFont.load_default()

    drawn_text_rects = []

    def draw_label(x, yc, is_custom=False):
        s_idx = np.argmin([abs(np.mean(s) - yc) for s in staves])
        clef = "treble" if s_idx % 2 == 0 else "bass"
        p_name = get_pitch_name(yc, staves[s_idx], clef)
        
        lbl_key = (int(x), int(yc))
        if lbl_key in custom_labels:
            p_name = custom_labels[lbl_key]
        
        if p_name or not hide_boxes:
            hw, hh = int(space * 0.6), int(space * 0.5)
            b = [x - hw, yc - hh, x + hw, yc + hh]
            
            if not hide_boxes:
                box_color = (255, 165, 0) if is_custom else (0, 255, 0)
                if selected_pos and math.hypot(x - selected_pos[0], yc - selected_pos[1]) < 2:
                    draw.rectangle(b, outline=(0, 0, 255), width=4)
                else:
                    draw.rectangle(b, outline=box_color, width=2)
            
            if p_name:
                color = (255, 0, 0)
                text_x = b[0]
                text_y = b[1] - int(space * 1.6)
                text_w = len(p_name) * font_size
                text_h = font_size
                
                while True:
                    is_overlapping = False
                    for (rx, ry, rw, rh) in drawn_text_rects:
                        if not (text_x + text_w < rx or text_x > rx + rw or text_y + text_h < ry or text_y > ry + rh):
                            is_overlapping = True
                            break
                    if is_overlapping:
                        text_x += int(font_size * 1.1)
                    else:
                        break

                draw.text((text_x, text_y), p_name, font=font, fill=color)
                drawn_text_rects.append((text_x, text_y, text_w, text_h))

    for box in auto_notes:
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        is_deleted = any(math.hypot(cx - dx, cy - dy) < 1.0 for dx, dy in deleted_auto)
        if not is_deleted:
            draw_label(cx, cy, False)

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
# 3. Streamlit UI (ウィザード形式)
# ==========================================
st.set_page_config(page_title="ドレミ付与 V8", layout="wide") 

if "step" not in st.session_state: st.session_state.step = 1
if "custom_clicks" not in st.session_state: st.session_state.custom_clicks = {}
if "deleted_auto_notes" not in st.session_state: st.session_state.deleted_auto_notes = {}
if "custom_labels" not in st.session_state: st.session_state.custom_labels = {} 
if "selected_note" not in st.session_state: st.session_state.selected_note = None 
if "pdf_data" not in st.session_state: st.session_state.pdf_data = None
if "ui_sens" not in st.session_state: st.session_state.ui_sens = 50 # スライダーの初期値

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1

st.title("🎼 ドレミ付与ツール V8")

steps = ["1. アップロード", "2. ワンクリック調整", "3. マニュアル微調整", "4. プレビュー＆保存"]
cols = st.columns(4)
for i, step_name in enumerate(steps):
    if st.session_state.step == i + 1:
        cols[i].markdown(f"**🔵 {step_name}**")
    else:
        cols[i].markdown(f"⚪ {step_name}")
st.divider()

# 表示サイズは内部で 800px に固定します
FIXED_DISP_WIDTH = 800 
# スライダーの値を元に検出閾値を計算
internal_threshold = 0.85 - (st.session_state.ui_sens / 100.0) * 0.40

# PDFの処理（キャッシュ化されているので閾値が変わった時だけ再計算されます）
if st.session_state.pdf_data:
    pages = process_pdf_and_detect(st.session_state.pdf_data, internal_threshold)
else:
    pages = []

# ==========================================
# STEP 1: アップロード
# ==========================================
if st.session_state.step == 1:
    st.subheader("Step 1: 楽譜PDFをアップロード")
    up = st.file_uploader("PDFファイルを選択してください", type="pdf")
    
    if up:
        st.session_state.pdf_data = up.getvalue()
        st.success("PDFを読み込みました！次へ進んでください。")
        st.button("次へ ➡️", on_click=next_step, type="primary")

# ==========================================
# STEP 2: ワンクリック自動調整
# ==========================================
if st.session_state.step == 2:
    col1, col2 = st.columns([1, 1])
    col1.subheader("Step 2: ワンクリック調整")
    col2.button("次へ：テキストの微調整 ➡️", on_click=next_step, type="primary")
    col2.button("⬅️ やり直す (Step 1へ)", on_click=prev_step)

    # ★感度スライダーをここに配置
    st.slider("🔍 自動検出の感度（ページを見ながら調整できます）", 1, 100, key="ui_sens")
    st.info("💡 **操作:** まずスライダーで良い感じの感度に合わせます。そのあと、検出漏れの場所をクリックで追加、既存の音符の近くをクリックで削除できます。")

    for i, page in enumerate(pages):
        st.write(f"### ページ {i + 1}")
        if page["staves"]:
            clicks = st.session_state.custom_clicks.setdefault(i, [])
            deleted_auto = st.session_state.deleted_auto_notes.setdefault(i, [])
            
            res_img = draw_all_notes(page["image"], page["notes"], clicks, deleted_auto, page["staves"], page["space"], st.session_state.custom_labels.get(i, {}))
            value = streamlit_image_coordinates(res_img, key=f"s2_img_{i}", width=FIXED_DISP_WIDTH)
            
            last_click_key = f"s2_last_click_{i}"
            if last_click_key not in st.session_state: st.session_state[last_click_key] = None
            
            if value and value != st.session_state[last_click_key]:
                st.session_state[last_click_key] = value
                scale = page["image"].width / FIXED_DISP_WIDTH
                real_x, real_y = value["x"] * scale, value["y"] * scale
                click_threshold = page["space"] * 1.5 
                action_taken = False

                for pt in clicks.copy():
                    if math.hypot(real_x - pt[0], real_y - pt[1]) < click_threshold:
                        clicks.remove(pt)
                        action_taken = True
                        break

                if not action_taken:
                    for box in page["notes"]:
                        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                        if not any(math.hypot(cx - dx, cy - dy) < 1.0 for dx, dy in deleted_auto):
                            if math.hypot(real_x - cx, real_y - cy) < click_threshold:
                                deleted_auto.append((cx, cy))
                                action_taken = True
                                break

                if not action_taken:
                    clicks.append((real_x, real_y))
                st.rerun() 
        else:
            st.warning("⚠️ このページからは五線が検出されませんでした。")
            st.image(page["image"], width=FIXED_DISP_WIDTH)

# ==========================================
# STEP 3: マニュアル微調整
# ==========================================
if st.session_state.step == 3:
    col1, col2 = st.columns([1, 1])
    col1.subheader("Step 3: マニュアル微調整")
    col1.info("💡 **操作:** 音符をクリックして選択し、下の入力欄で「ドレミ」を自由に変更できます。")
    col2.button("次へ：完成プレビュー ➡️", on_click=next_step, type="primary")
    col2.button("⬅️ 戻る (Step 2へ)", on_click=prev_step)

    for i, page in enumerate(pages):
        st.write(f"### ページ {i + 1}")
        if page["staves"]:
            clicks = st.session_state.custom_clicks.get(i, [])
            deleted_auto = st.session_state.deleted_auto_notes.get(i, [])
            custom_labels_page = st.session_state.custom_labels.setdefault(i, {})
            
            sel_pos = None
            if st.session_state.selected_note and st.session_state.selected_note["page"] == i:
                sel_pos = (st.session_state.selected_note["x"], st.session_state.selected_note["y"])

            res_img = draw_all_notes(page["image"], page["notes"], clicks, deleted_auto, page["staves"], page["space"], custom_labels_page, selected_pos=sel_pos)
            value = streamlit_image_coordinates(res_img, key=f"s3_img_{i}", width=FIXED_DISP_WIDTH)
            
            last_click_key = f"s3_last_click_{i}"
            if last_click_key not in st.session_state: st.session_state[last_click_key] = None
            
            if value and value != st.session_state[last_click_key]:
                st.session_state[last_click_key] = value
                scale = page["image"].width / FIXED_DISP_WIDTH
                real_x, real_y = value["x"] * scale, value["y"] * scale
                click_threshold = page["space"] * 1.5 
                
                found_note = None
                for pt in clicks:
                    if math.hypot(real_x - pt[0], real_y - pt[1]) < click_threshold:
                        found_note = (int(pt[0]), int(pt[1]))
                        break
                if not found_note:
                    for box in page["notes"]:
                        cx, cy = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)
                        if not any(math.hypot(cx - dx, cy - dy) < 1.0 for dx, dy in deleted_auto):
                            if math.hypot(real_x - cx, real_y - cy) < click_threshold:
                                found_note = (cx, cy)
                                break
                
                if not found_note:
                    clicks.append((real_x, real_y))
                    found_note = (int(real_x), int(real_y))
                
                current_label = custom_labels_page.get(found_note)
                if current_label is None:
                    s_idx = np.argmin([abs(np.mean(s) - found_note[1]) for s in page["staves"]])
                    clef = "treble" if s_idx % 2 == 0 else "bass"
                    current_label = get_pitch_name(found_note[1], page["staves"][s_idx], clef)
                
                st.session_state.selected_note = {"page": i, "x": found_note[0], "y": found_note[1], "label": current_label}
                st.rerun()

            if st.session_state.selected_note and st.session_state.selected_note["page"] == i:
                sel = st.session_state.selected_note
                new_label = st.text_input(f"✎ 選択中の音符のテキスト（ページ {i+1}）", value=sel["label"], key=f"input_{i}")
                if new_label != sel["label"]:
                    custom_labels_page[(sel["x"], sel["y"])] = new_label
                    st.session_state.selected_note["label"] = new_label
                    st.rerun()
        else:
            st.image(page["image"], width=FIXED_DISP_WIDTH)

# ==========================================
# STEP 4: プレビュー＆ダウンロード
# ==========================================
if st.session_state.step == 4:
    st.subheader("Step 4: 完成プレビュー＆ダウンロード")
    col1, col2 = st.columns([1, 1])
    col1.info("✅ 枠線が消え、テキストのみが描画された完成版です。")
    col2.button("⬅️ 戻る (Step 3へ)", on_click=prev_step)
    
    out_images = []
    for i, page in enumerate(pages):
        st.write(f"### ページ {i + 1}")
        if page["staves"]:
            res_img = draw_all_notes(
                page["image"], 
                page["notes"], 
                st.session_state.custom_clicks.get(i, []), 
                st.session_state.deleted_auto_notes.get(i, []), 
                page["staves"], 
                page["space"], 
                st.session_state.custom_labels.get(i, {}),
                hide_boxes=True
            )
            out_images.append(res_img)
            st.image(res_img, width=FIXED_DISP_WIDTH)
        else:
            out_images.append(page["image"])
            st.image(page["image"], width=FIXED_DISP_WIDTH)
            
    if out_images:
        pdf_buffer = io.BytesIO()
        out_images[0].save(pdf_buffer, format="PDF", save_all=True, append_images=out_images[1:])
        st.download_button(
            label="📥 楽譜をPDFでダウンロード",
            data=pdf_buffer.getvalue(),
            file_name="score_with_notes.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

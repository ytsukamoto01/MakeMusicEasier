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

    # --- 太い線（連桁や太い縦線）の感知 ---
    beam_w = int(staff_space * 1.5)
    beam_h = max(2, int(staff_space * 0.25)) 
    beam_k = cv2.getStructuringElement(cv2.MORPH_RECT, (beam_w, beam_h))
    thick_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, beam_k)

    v_beam_w = max(2, int(staff_space * 0.25))
    v_beam_h = int(staff_space * 2.5)
    v_beam_k = cv2.getStructuringElement(cv2.MORPH_RECT, (v_beam_w, v_beam_h))
    thick_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_beam_k)

    thick_lines_mask = cv2.bitwise_or(thick_horizontal, thick_vertical)
    thresh_for_notes = cv2.subtract(thresh, thick_lines_mask)

    # --- 符幹の切断 ---
    sever_k_size = max(2, int(staff_space * 0.35))
    sever_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sever_k_size, sever_k_size))
    severed = cv2.morphologyEx(thresh_for_notes, cv2.MORPH_OPEN, sever_k)

    num_labels, labels_sev, stats_sev, _ = cv2.connectedComponentsWithStats(severed, connectivity=8)
    notes_only = np.zeros_like(thresh_for_notes)

    for label in range(1, num_labels):
        w = stats_sev[label, cv2.CC_STAT_WIDTH]
        h = stats_sev[label, cv2.CC_STAT_HEIGHT]
        area = stats_sev[label, cv2.CC_STAT_AREA]
        if w > staff_space * 2.0 or h > staff_space * 4.0 or area > staff_space**2 * 3.0 or area < staff_space**2 * 0.4:
            continue
        notes_only[labels_sev == label] = 255

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(2, int(staff_space * 0.4)), max(2, int(staff_space * 0.4))))
    filled = cv2.morphologyEx(notes_only, cv2.MORPH_CLOSE, close_k)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled, connectivity=8)

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
        cx, cy = x + w // 2, y + h // 2
        dist_to_nearest_staff = min(abs(cy - c) for c in staff_centers)
        if dist_to_nearest_staff > staff_space * 3.0 or cy >= labels.shape[0] or cx >= labels.shape[1]:
            continue
        
        label_id = labels[cy, cx]
        if label_id == 0: continue
        if stats[label_id, cv2.CC_STAT_AREA] < staff_space**2 * 0.4: continue

        raw_rects.append([x, y, x + w, y + h])
        raw_scores.append(score)

    if not raw_rects:
        return [], thick_lines_mask

    nms_boxes = nms_v8_strict(np.array(raw_rects), np.array(raw_scores), staff_space)
    
    # 幹チェック
    final_boxes = []
    stem_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(staff_space * 0.6))))
    for box in nms_boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        w = x2 - x1
        stem_check_h = int(staff_space * 2.0)
        
        # 上下どちらかに幹があるか
        has_stem = False
        for (sy1, sy2) in [(max(0, y1-stem_check_h), y1), (y2, min(thresh.shape[0], y2+stem_check_h))]:
            if sy2 > sy1:
                patch = thresh[sy1:sy2, max(0, cx-2):min(thresh.shape[1], cx+2)]
                if np.sum(cv2.morphologyEx(patch, cv2.MORPH_OPEN, stem_k)) > 0:
                    has_stem = True
                    break
        if has_stem: final_boxes.append(box)

    return np.array(final_boxes) if final_boxes else [], thick_lines_mask

def get_pitch_name(note_y, staff, clef):
    line1, line5 = staff[0], staff[4]
    step_size = abs(line1 - line5) / 8.0
    steps = int(round((line1 - note_y) / step_size))
    if clef == "treble":
        mapping = {-4:"ラ",-3:"シ",-2:"ド",-1:"レ",0:"ミ",1:"ファ",2:"ソ",3:"ラ",4:"シ",5:"ド",6:"レ",7:"ミ",8:"ファ",9:"ソ",10:"ラ",11:"シ",12:"ド",13:"レ",14:"ミ",15:"ファ",16:"ソ"}
    else:
        mapping = {-6:"ラ",-5:"シ",-4:"ド",-3:"レ",-2:"ミ",-1:"ファ",0:"ソ",1:"ラ",2:"シ",3:"ド",4:"レ",5:"ミ",6:"ファ",7:"ソ",8:"ラ",9:"シ",10:"ド",11:"レ",12:"ミ",13:"ファ",14:"ソ",15:"ラ",16:"シ"}
    return mapping.get(steps, "")

# ==========================================
# 2. 描画・ユーティリティ
# ==========================================

def draw_all_notes(pil_img, auto_notes, custom_clicks, deleted_auto, staves, space, custom_labels, hide_boxes=False, selected_pos=None, erase_start=None, beams_mask=None):
    result = pil_img.copy().convert("RGBA")
    
    # デバッグ用：太い線をマゼンタで表示
    if beams_mask is not None and not hide_boxes:
        color_layer = Image.new("RGBA", result.size, (255, 0, 255, 128))
        mask_img = Image.fromarray(beams_mask).convert("L")
        result.paste(color_layer, (0, 0), mask=mask_img)

    result = result.convert("RGB")
    draw = ImageDraw.Draw(result)
    
    font_size = max(15, int(space))
    try:
        font = ImageFont.truetype("NotoSansJP-Regular.ttf", font_size)
    except:
        font = ImageFont.load_default()

    active_notes = []
    for box in auto_notes:
        cx, cy = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)
        if not any(math.hypot(cx - dx, cy - dy) < 2.0 for dx, dy in deleted_auto):
            s_idx = np.argmin([abs(np.mean(s) - cy) for s in staves])
            active_notes.append({"x": cx, "y": cy, "s_idx": s_idx, "is_custom": False})
            
    for cx, cy in custom_clicks:
        s_idx = np.argmin([abs(np.mean(s) - cy) for s in staves])
        active_notes.append({"x": int(cx), "y": int(cy), "s_idx": s_idx, "is_custom": True})

    active_notes.sort(key=lambda n: (n["s_idx"], n["x"]))
    
    groups = []
    for note in active_notes:
        if not groups: groups.append([note])
        else:
            last = groups[-1]
            if abs(note["x"] - (sum(n["x"] for n in last)/len(last))) < space * 0.8 and note["s_idx"] == last[0]["s_idx"]:
                last.append(note)
            else: groups.append([note])

    drawn_text_rects = []
    for group in groups:
        group.sort(key=lambda n: n["y"])
        labels_to_draw = []
        for note in group:
            x, y, s_idx = note["x"], note["y"], note["s_idx"]
            p_name = None
            for (lx, ly), label in custom_labels.items():
                if math.hypot(x - lx, y - ly) < 2.0:
                    p_name = label
                    break
            if p_name is None:
                p_name = get_pitch_name(y, staves[s_idx], "treble" if s_idx % 2 == 0 else "bass")
            
            if p_name: labels_to_draw.append(p_name)
            if not hide_boxes:
                box_color = (255, 165, 0) if note["is_custom"] else (0, 255, 0)
                hw, hh = int(space * 0.6), int(space * 0.5)
                if selected_pos and math.hypot(x - selected_pos[0], y - selected_pos[1]) < 2:
                    draw.rectangle([x-hw, y-hh, x+hw, y+hh], outline=(0, 0, 255), width=4)
                else:
                    draw.rectangle([x-hw, y-hh, x+hw, y+hh], outline=box_color, width=2)

        if labels_to_draw:
            combined_text = " ".join(labels_to_draw)
            tx, ty = group[0]["x"] - int(space*0.6), group[0]["y"] - int(space*1.6)
            draw.text((tx, ty), combined_text, font=font, fill=(255, 0, 0))

    if erase_start and not hide_boxes:
        ex, ey = erase_start
        draw.ellipse([ex-5, ey-5, ex+5, ey+5], fill=(255, 0, 0))
    return result

@st.cache_data(show_spinner=False)
def process_pdf_and_detect(pdf_bytes, threshold):
    imgs = convert_from_bytes(pdf_bytes)
    data = []
    for img in imgs:
        staves, space = detect_staff_groups_v8(img)
        notes, mask = detect_note_heads_v8(np.array(img.convert('L')), space, threshold, staves) if staves else ([], None)
        data.append({"image": img, "staves": staves, "space": space, "notes": notes, "beams_mask": mask})
    return data

# ==========================================
# 3. Streamlit UI
# ==========================================

st.set_page_config(page_title="ドレミ付与 V8", layout="wide")

for key in ["step", "custom_clicks", "deleted_auto_notes", "custom_labels", "selected_note", "pdf_data"]:
    if key not in st.session_state:
        if key == "step": st.session_state[key] = 1
        elif "data" in key: st.session_state[key] = None
        else: st.session_state[key] = {}

st.title("🎼 ドレミ付与ツール V8")
st.divider()

FIXED_WIDTH = 800
internal_threshold = 0.45

if st.session_state.step == 1:
    up = st.file_uploader("PDFファイルを選択", type="pdf")
    if up:
        st.session_state.pdf_data = up.getvalue()
        st.button("次へ ➡️", on_click=lambda: setattr(st.session_state, 'step', 2))

elif st.session_state.step >= 2:
    pages = process_pdf_and_detect(st.session_state.pdf_data, internal_threshold)
    
    if st.session_state.step == 2:
        col1, col2 = st.columns([3, 1])
        with col2:
            st.button("微調整へ ➡️", on_click=lambda: setattr(st.session_state, 'step', 3), type="primary")
            edit_mode = st.radio("モード", ["通常", "範囲消去"])
        
        with col1:
            for i, page in enumerate(pages):
                st.write(f"ページ {i+1}")
                clicks = st.session_state.custom_clicks.setdefault(i, [])
                deleted = st.session_state.deleted_auto_notes.setdefault(i, [])
                
                res_img = draw_all_notes(page["image"], page["notes"], clicks, deleted, page["staves"], page["space"], {}, beams_mask=page["beams_mask"])
                val = streamlit_image_coordinates(res_img, width=FIXED_WIDTH, key=f"img_{i}")
                
                if val:
                    scale = page["image"].width / FIXED_WIDTH
                    rx, ry = val["x"] * scale, val["y"] * scale
                    # 簡易クリック判定（本来はここで再描画ロジックを回す）
                    if not any(math.hypot(rx-c[0], ry-c[1]) < 20 for c in clicks):
                        clicks.append((rx, ry))
                        st.rerun()

    elif st.session_state.step == 3:
        if st.button("⬅️ 戻る"): st.session_state.step = 2
        st.button("プレビュー ➡️", on_click=lambda: setattr(st.session_state, 'step', 4))
        for i, page in enumerate(pages):
            st.image(draw_all_notes(page["image"], page["notes"], st.session_state.custom_clicks.get(i,[]), st.session_state.deleted_auto_notes.get(i,[]), page["staves"], page["space"], st.session_state.custom_labels.get(i,{}), beams_mask=page["beams_mask"]), width=FIXED_WIDTH)

    elif st.session_state.step == 4:
        st.button("⬅️ 戻る", on_click=lambda: setattr(st.session_state, 'step', 3))
        out_imgs = []
        for i, page in enumerate(pages):
            res = draw_all_notes(page["image"], page["notes"], st.session_state.custom_clicks.get(i,[]), st.session_state.deleted_auto_notes.get(i,[]), page["staves"], page["space"], st.session_state.custom_labels.get(i,{}), hide_boxes=True)
            out_imgs.append(res)
            st.image(res, width=FIXED_WIDTH)
        
        buf = io.BytesIO()
        out_imgs[0].save(buf, format="PDF", save_all=True, append_images=out_imgs[1:])
        st.download_button("PDFダウンロード", buf.getvalue(), "score.pdf", "application/pdf")

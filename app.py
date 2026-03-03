import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import math
from pdf2image import convert_from_bytes
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. 楽譜解析エンジン (V8 ロジック + 線分マスク)
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
    
    # ===== [NEW] 線分検出によるマスク作成 =====
    # LSD (Line Segment Detector) を初期化
    lsd = cv2.createLineSegmentDetector(0)
    lines, width, prec, nfa = lsd.detect(gray_img)
    
    # マスク画像（白地に黒い線分領域）を作成
    line_mask = np.ones_like(gray_img) * 255  # 初期は全白
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 線分の長さが staff_space の 2.5 倍以上あるものを「大きな線」とみなす
            length = np.hypot(x2 - x1, y2 - y1)
            if length > staff_space * 2.5:
                # 線分を太らせてマスクに描画（太さは staff_space の 0.4 倍）
                thickness = max(2, int(staff_space * 0.4))
                cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, thickness)
    # マスクを二値化（線分領域 = 0, それ以外 = 255）
    _, line_mask = cv2.threshold(line_mask, 127, 255, cv2.THRESH_BINARY)
    # ========================================

    open_k_size = max(3, int(staff_space * 0.6))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k_size, open_k_size))
    notes_only = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
    close_k_size = max(3, int(staff_space * 0.3))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k_size, close_k_size))
    filled = cv2.morphologyEx(notes_only, cv2.MORPH_CLOSE, close_k)

    # 画像内の「黒い塊」ごとの大きさを計測
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
        cx, cy = x + w//2, y + h//2
        
        # ===== [NEW] 線分マスク上でこの点が有効かチェック =====
        if cy >= line_mask.shape[0] or cx >= line_mask.shape[1]:
            continue
        if line_mask[cy, cx] == 0:   # 線分領域内なら除外
            continue
        # ==================================================
        
        dist_to_nearest_staff = min(abs(cy - c) for c in staff_centers)
        if dist_to_nearest_staff > staff_space * 4.0: continue
        
        if cy >= labels.shape[0] or cx >= labels.shape[1]: continue
        label = labels[cy, cx]
        if label == 0: continue
        
        comp_w = stats[label, cv2.CC_STAT_WIDTH]
        comp_h = stats[label, cv2.CC_STAT_HEIGHT]
        comp_area = stats[label, cv2.CC_STAT_AREA]
        
        if comp_w > staff_space * 3.5: continue
        if comp_h > staff_space * 5.5: continue
        if comp_area > staff_space ** 2 * 3.0: continue

        patch = filled[y:y+h, x:x+w]
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            
            if area < (staff_space**2) * 0.3: continue
            
            rect = cv2.minAreaRect(cnt)
            (rw, rh) = rect[1]
            if rw == 0 or rh == 0: continue
            
            rotated_aspect = max(rw, rh) / min(rw, rh)
            if rotated_aspect > 2.0: continue 
            
            rotated_extent = area / (rw * rh)
            if rotated_extent > 0.85: continue 
            
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circularity = 4.0 * np.pi * area / (peri**2)
                if circularity >= 0.65: 
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = area / float(hull_area)
                        if solidity >= 0.85:
                            raw_rects.append([x, y, x+w, y+h])
                            raw_scores.append(score)
    
    nms_boxes = nms_v8_strict(np.array(raw_rects), np.array(raw_scores), staff_space) if raw_rects else []
    
    if len(nms_boxes) == 0: return []

    final_boxes = []
    stem_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(staff_space * 0.5))))
    stem_check_h = int(staff_space * 1.5)
    stem_check_w_ratio = 0.3 

    for box in nms_boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1
        cx = (x1 + x2) // 2
        
        stem_up_x1 = max(0, cx - int(w * stem_check_w_ratio / 2))
        stem_up_x2 = min(thresh.shape[1], cx + int(w * stem_check_w_ratio / 2))
        stem_up_y1 = max(0, y1 - stem_check_h)
        stem_up_y2 = y1
        
        stem_down_x1 = stem_up_x1
        stem_down_x2 = stem_up_x2
        stem_down_y1 = y2
        stem_down_y2 = min(thresh.shape[0], y2 + stem_check_h)
        
        has_stem = False
        if stem_up_y2 > stem_up_y1 and stem_up_x2 > stem_up_x1:
            opened_up = cv2.morphologyEx(thresh[stem_up_y1:stem_up_y2, stem_up_x1:stem_up_x2], cv2.MORPH_OPEN, stem_k)
            if np.sum(opened_up) > 0: has_stem = True
                
        if stem_down_y2 > stem_down_y1 and stem_down_x2 > stem_down_x1:
            opened_down = cv2.morphologyEx(thresh[stem_down_y1:stem_down_y2, stem_down_x1:stem_down_x2], cv2.MORPH_OPEN, stem_k)
            if np.sum(opened_down) > 0: has_stem = True
                
        if has_stem: final_boxes.append(box)

    return np.array(final_boxes) if final_boxes else []

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
def draw_all_notes(pil_img, auto_notes, custom_clicks, deleted_auto, staves, space, custom_labels, hide_boxes=False, selected_pos=None, erase_start=None):
    result = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    
    font_size = max(15, int(space))
    try:
        font = ImageFont.truetype("NotoSansJP-Regular.ttf", font_size)
    except:
        font = ImageFont.load_default()

    drawn_text_rects = []

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
        if not groups:
            groups.append([note])
        else:
            last_group = groups[-1]
            avg_x = sum(n["x"] for n in last_group) / len(last_group)
            if abs(note["x"] - avg_x) < space * 0.8 and note["s_idx"] == last_group[0]["s_idx"]:
                last_group.append(note)
            else:
                groups.append([note])

    for group in groups:
        group.sort(key=lambda n: n["y"]) 
        labels_to_draw = []
        
        for note in group:
            x, y, s_idx, is_custom = note["x"], note["y"], note["s_idx"], note["is_custom"]
            clef = "treble" if s_idx % 2 == 0 else "bass"
            
            p_name = None
            for (lx, ly), label in custom_labels.items():
                if math.hypot(x - lx, y - ly) < 2.0:
                    p_name = label
                    break
            if p_name is None:
                p_name = get_pitch_name(y, staves[s_idx], clef)
                
            if p_name:
                labels_to_draw.append(p_name)
                
            if not hide_boxes:
                box_color = (255, 165, 0) if is_custom else (0, 255, 0)
                hw, hh = int(space * 0.6), int(space * 0.5)
                b = [x - hw, y - hh, x + hw, y + hh]
                
                if selected_pos and math.hypot(x - selected_pos[0], y - selected_pos[1]) < 2:
                    draw.rectangle(b, outline=(0, 0, 255), width=4)
                else:
                    draw.rectangle(b, outline=box_color, width=2)

        if labels_to_draw and (not hide_boxes or any(l for l in labels_to_draw)):
            min_y = group[0]["y"]
            group_avg_x = sum(n["x"] for n in group) / len(group)
            
            combined_text = " ".join(labels_to_draw)
            
            text_x = group_avg_x - int(space * 0.6)
            text_y = min_y - int(space * 1.6)
            text_w = len(combined_text) * font_size
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
                    
            draw.text((text_x, text_y), combined_text, font=font, fill=(255, 0, 0))
            drawn_text_rects.append((text_x, text_y, text_w, text_h))

    if erase_start and not hide_boxes:
        ex, ey = erase_start
        r = max(4, int(space * 0.4))
        draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(255, 0, 0))
        draw.line((ex, 0, ex, result.height), fill=(255, 0, 0), width=1)
        draw.line((0, ey, result.width, ey), fill=(255, 0, 0), width=1)
        
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

if "step" not in st.session_state: st.session_state.step = 1
if "custom_clicks" not in st.session_state: st.session_state.custom_clicks = {}
if "deleted_auto_notes" not in st.session_state: st.session_state.deleted_auto_notes = {}
if "custom_labels" not in st.session_state: st.session_state.custom_labels = {} 
if "selected_note" not in st.session_state: st.session_state.selected_note = None 
if "pdf_data" not in st.session_state: st.session_state.pdf_data = None
if "ui_sens" not in st.session_state: st.session_state.ui_sens = 50 

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

FIXED_DISP_WIDTH = 800 
internal_threshold = 100

if st.session_state.pdf_data:
    pages = process_pdf_and_detect(st.session_state.pdf_data, internal_threshold)
else:
    pages = []

if st.session_state.step == 1:
    st.subheader("Step 1: 楽譜PDFをアップロード")
    up = st.file_uploader("PDFファイルを選択してください", type="pdf")
    if up:
        st.session_state.pdf_data = up.getvalue()
        st.success("PDFを読み込みました！次へ進んでください。")
        st.button("次へ ➡️", on_click=next_step, type="primary")

if st.session_state.step == 2:
    subheader_col1, subheader_col2 = st.columns([2, 1])
    subheader_col1.subheader("Step 2: 自動検出の調整")
    subheader_col2.button("次へ：テキストの微調整 ➡️", on_click=next_step, type="primary")
    subheader_col2.button("⬅️ やり直す (Step 1へ)", on_click=prev_step)

    img_container_col, slider_container_col = st.columns([4, 1]) 

    with slider_container_col:
        st.write("### ") 
        st.subheader("⚙️ 調整設定")
        st.slider("🔍 検出感度", 1, 100, key="ui_sens")
        st.divider()
        st.subheader("🖱️ 操作モード")
        edit_mode = st.radio("画像クリック時の動作", ["👆 通常\n(追加 / 個別削除)", "🔲 範囲消去\n(2点クリックで一括削除)"])
        if "範囲消去" in edit_mode:
            st.warning("⚠️ **範囲消去モード中**\n消したいエリアの「左上」をクリックし、次に「右下」をクリックしてください。")

    with img_container_col:
        for i, page in enumerate(pages):
            st.write(f"### ページ {i + 1}")
            if page["staves"]:
                clicks = st.session_state.custom_clicks.setdefault(i, [])
                deleted_auto = st.session_state.deleted_auto_notes.setdefault(i, [])
                
                erase_start_key = f"s2_erase_start_{i}"
                if erase_start_key not in st.session_state: 
                    st.session_state[erase_start_key] = None
                
                if "範囲消去" not in edit_mode:
                    st.session_state[erase_start_key] = None

                res_img = draw_all_notes(
                    page["image"], page["notes"], clicks, deleted_auto, page["staves"], page["space"], 
                    st.session_state.custom_labels.get(i, {}), erase_start=st.session_state[erase_start_key]
                )
                
                value = streamlit_image_coordinates(res_img, key=f"s2_img_{i}", width=FIXED_DISP_WIDTH)
                
                last_click_key = f"s2_last_click_{i}"
                if last_click_key not in st.session_state: st.session_state[last_click_key] = None
                
                if value and value != st.session_state[last_click_key]:
                    st.session_state[last_click_key] = value 
                    
                    clicked_x, clicked_y = value["x"], value["y"]
                    scale = page["image"].width / FIXED_DISP_WIDTH
                    real_x, real_y = clicked_x * scale, clicked_y * scale
                    
                    if "範囲消去" in edit_mode:
                        if st.session_state[erase_start_key] is None:
                            st.session_state[erase_start_key] = (real_x, real_y)
                        else:
                            x1, y1 = st.session_state[erase_start_key]
                            x2, y2 = real_x, real_y
                            min_x, max_x, min_y, max_y = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
                            clicks[:] = [pt for pt in clicks if not (min_x <= pt[0] <= max_x and min_y <= pt[1] <= max_y)]
                            for box in page["notes"]:
                                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                                if min_x <= cx <= max_x and min_y <= cy <= max_y:
                                    if not any(math.hypot(cx - dx, cy - dy) < 2.0 for dx, dy in deleted_auto):
                                        deleted_auto.append((cx, cy))
                            st.session_state[erase_start_key] = None
                    else:
                        hit_threshold_x = page["space"] * 0.8
                        hit_threshold_y = page["space"] * 0.45 
                        action_taken = False

                        for pt in clicks.copy():
                            if abs(real_x - pt[0]) < hit_threshold_x and abs(real_y - pt[1]) < hit_threshold_y:
                                clicks.remove(pt)
                                action_taken = True
                                break

                        if not action_taken:
                            for box in page["notes"]:
                                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                                if not any(math.hypot(cx - dx, cy - dy) < 2.0 for dx, dy in deleted_auto):
                                    if abs(real_x - cx) < hit_threshold_x and abs(real_y - cy) < hit_threshold_y:
                                        deleted_auto.append((cx, cy))
                                        action_taken = True
                                        break

                        if not action_taken:
                            clicks.append((real_x, real_y))
                    st.rerun() 
            else:
                st.warning(f"⚠️ ページ {i+1} からは五線が検出されませんでした。")
                st.image(page["image"], width=FIXED_DISP_WIDTH)

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
                
                hit_threshold_x = page["space"] * 0.8
                hit_threshold_y = page["space"] * 0.45 
                
                found_note = None
                for pt in clicks:
                    if abs(real_x - pt[0]) < hit_threshold_x and abs(real_y - pt[1]) < hit_threshold_y:
                        found_note = (int(pt[0]), int(pt[1]))
                        break
                if not found_note:
                    for box in page["notes"]:
                        cx, cy = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)
                        if not any(math.hypot(cx - dx, cy - dy) < 2.0 for dx, dy in deleted_auto):
                            if abs(real_x - cx) < hit_threshold_x and abs(real_y - cy) < hit_threshold_y:
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
                page["image"], page["notes"], st.session_state.custom_clicks.get(i, []), 
                st.session_state.deleted_auto_notes.get(i, []), page["staves"], page["space"], 
                st.session_state.custom_labels.get(i, {}), hide_boxes=True
            )
            out_images.append(res_img)
            st.image(res_img, width=FIXED_DISP_WIDTH)
        else:
            out_images.append(page["image"])
            st.image(page["image"], width=FIXED_DISP_WIDTH)
            
    if out_images:
        pdf_buffer = io.BytesIO()
        out_images[0].save(pdf_buffer, format="PDF", save_all=True, append_images=out_images[1:])
        st.download_button("📥 楽譜をPDFでダウンロード", data=pdf_buffer.getvalue(), file_name="score_with_notes.pdf", mime="application/pdf", type="primary", use_container_width=True)

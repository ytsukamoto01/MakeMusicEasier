import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import math
from pdf2image import convert_from_bytes
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. 高精度楽譜解析エンジン (V9)
# ==========================================
def detect_staff_lines_hough(gray_img):
    """Hough変換で5線を高精度検出"""
    edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=gray_img.shape[1]*0.7, maxLineGap=10)
    
    if lines is None:
        return []
    
    # 水平線のみ抽出（角度±5度以内）
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
        if abs(angle) < 5:  # ほぼ水平
            horizontal_lines.append((min(y1,y2), max(y1,y2)))
    
    # Y座標でソート・クラスタリング
    y_coords = sorted([y for _, y in horizontal_lines])
    clustered = []
    if y_coords:
        current = [y_coords[0]]
        for y in y_coords[1:]:
            if y - current[-1] < 20:  # 近接ピクセルは同一線
                current[-1] = (current[-1] + y) / 2
            else:
                clustered.append(current[-1])
                current = [y]
        clustered.append(current[-1])
    
    # 5本の線が等間隔か判定
    if len(clustered) >= 5:
        diffs = np.diff(clustered[:5])
        avg_space = np.mean(diffs)
        if np.all(np.abs(diffs - avg_space) < avg_space * 0.4):
            return clustered[:5], avg_space
    return [], 12

def detect_staff_groups_v9(pil_img):
    """改良版5線譜グループ検出"""
    gray = np.array(pil_img.convert('L'))
    
    # 複数スケールで5線検出
    staves1, space1 = detect_staff_lines_hough(gray)
    if not staves1:
        # フォールバック：投影法
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        proj = np.sum(thresh, axis=1)
        peaks = []
        thresh_val = np.max(proj) * 0.25
        for i in range(2, len(proj)-2):
            if proj[i] > thresh_val and proj[i] > max(proj[i-2:i]) and proj[i] > max(proj[i+1:i+3]):
                peaks.append(i)
        
        if len(peaks) >= 5:
            staves1, space1 = peaks[:5], np.mean(np.diff(peaks[:5]))
    
    # 縦方向のスタッフグループ化
    all_staves = []
    staff_height = int(space1 * 5 * 1.2)
    for i in range(gray.shape[0] - staff_height):
        window_lines = [line for line in staves1 if i <= line <= i + staff_height]
        if len(window_lines) >= 5:
            all_staves.append(sorted(window_lines[:5], reverse=True))
    
    # 重複除去
    unique_staves = []
    for staff in all_staves:
        if not any(np.all(np.abs(np.array(staff) - np.array(s)) < space1*0.5) for s in unique_staves):
            unique_staves.append(staff)
    
    return unique_staves, space1 if unique_staves else 12

def remove_staff_lines_and_noise(gray_img, staves, staff_space):
    """5線・ノイズを徹底除去"""
    h, w = gray_img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 5線をマスク
    for staff in staves:
        for line_y in staff:
            y1 = max(0, int(line_y - staff_space * 0.15))
            y2 = min(h, int(line_y + staff_space * 0.15))
            mask[y1:y2, :] = 255
    
    # 細いノイズを除去
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_small, iterations=1)
    
    # 連桁・太い線を追加除去
    thick_h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(staff_space*3), 3))
    thick_v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, int(staff_space*2)))
    thick_lines = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, thick_h_kernel)
    thick_lines = cv2.bitwise_or(thick_lines, cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, thick_v_kernel))
    cleaned = cv2.bitwise_or(cleaned, thick_lines)
    
    return cleaned

def detect_note_heads_v9(gray_img, staff_space, threshold_val=0.75, staves=None):
    """高精度符頭検出"""
    # 適応閾値で二値化
    thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 21, 12)
    
    # 5線・ノイズ除去
    if staves:
        noise_mask = remove_staff_lines_and_noise(thresh, staves, staff_space)
    else:
        noise_mask = thresh
    
    # 符頭サイズの楕円テンプレート（複数サイズ）
    templates = []
    sizes = [0.7, 0.9, 1.1]
    for scale in sizes:
        nw = int(staff_space * scale)
        nh = int(staff_space * scale * 0.8)
        temp = np.zeros((nh, nw), dtype=np.uint8)
        cv2.ellipse(temp, (nw//2, nh//2), (nw//2-1, nh//2-1), 0, 0, 360, 255, -1)
        templates.append(temp)
    
    # 複数テンプレートでマッチング
    matches = []
    for temp in templates:
        res = cv2.matchTemplate(noise_mask, temp, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold_val)
        for pt in zip(*loc[::-1]):
            matches.append((pt[0] + temp.shape[1]//2, pt[1] + temp.shape[0]//2, res[pt[1], pt[0]]))
    
    # NMS + 形状フィルタ
    if not matches:
        return np.array([])
    
    boxes = []
    for cx, cy, score in matches:
        # ROI抽出
        roi_size = int(staff_space * 1.5)
        x1, y1 = max(0, cx-roi_size//2), max(0, cy-roi_size//2)
        x2, y2 = min(gray_img.shape[1], cx+roi_size//2), min(gray_img.shape[0], cy+roi_size//2)
        roi = noise_mask[y1:y2, x1:x2]
        
        if roi.size == 0:
            continue
            
        # 輪郭解析
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        
        # 厳格な形状フィルタ
        if area < (staff_space**2 * 0.25) or area > (staff_space**2 * 2.5):
            continue
            
        rect = cv2.minAreaRect(cnt)
        (rw, rh), angle = rect[1], rect[2]
        aspect = max(rw, rh) / min(rw, rh)
        
        peri = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
        hull = cv2.convexHull(cnt)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        
        # 符頭の厳格条件
        if (0.6 <= circularity <= 0.95 and 
            0.8 <= solidity <= 0.98 and 
            0.6 <= aspect <= 2.0):
            
            boxes.append([cx-staff_space//3, cy-staff_space//3, 
                         cx+staff_space//3, cy+staff_space//3])
    
    # スタッフ範囲外を除外
    if staves:
        staff_centers = [np.mean(s) for s in staves]
        final_boxes = []
        for box in boxes:
            cy = (box[1] + box[3]) / 2
            if min(abs(cy - c) for c in staff_centers) < staff_space * 4:
                final_boxes.append(box)
        boxes = final_boxes
    
    return np.array(boxes) if boxes else np.array([])

def has_stem_v9(thresh, box, staff_space):
    """改良版幹検出（方向性考慮）"""
    x1, y1, x2, y2 = box.astype(int)
    w, h = x2 - x1, y2 - y1
    cx = (x1 + x2) // 2
    
    # 細長い縦線検出用カーネル
    stem_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w//4), max(3, int(staff_space*0.8))))
    
    # 上方向チェック
    stem_up = thresh[max(0, y1-int(staff_space*2)):y1, max(0, cx-w//6):min(thresh.shape[1], cx+w//6)]
    if stem_up.size > 0:
        stem_up_clean = cv2.morphologyEx(stem_up, cv2.MORPH_OPEN, stem_kernel)
        if np.sum(stem_up_clean) > stem_up.size * 0.1:
            return True
    
    # 下方向チェック
    stem_down = thresh[y2:min(thresh.shape[0], y2+int(staff_space*2)), max(0, cx-w//6):min(thresh.shape[1], cx+w//6)]
    if stem_down.size > 0:
        stem_down_clean = cv2.morphologyEx(stem_down, cv2.MORPH_OPEN, stem_kernel)
        if np.sum(stem_down_clean) > stem_down.size * 0.1:
            return True
    
    return False

def get_pitch_name(note_y, staff, clef):
    """音名取得（改良版）"""
    line_positions = staff  # 上から line1, line2, line3, line4, line5
    space_positions = []
    for i in range(4):
        space_positions.append((line_positions[i] + line_positions[i+1]) / 2)
    
    all_positions = line_positions + space_positions
    all_positions.sort()
    
    # 最も近い線/間を特定
    distances = [abs(note_y - pos) for pos in all_positions]
    closest_idx = np.argmin(distances)
    
    if clef == "treble":
        names = ["ファ", "ミ", "レ", "ド", "シ", "ラ", "ソ", "ファ", "ミ", "レ", "ド", "シ", "ラ", "ソ"]
    else:  # bass
        names = ["シ", "ラ", "ソ", "ファ", "ミ", "レ", "ド", "シ", "ラ", "ソ", "ファ", "ミ", "レ", "ド"]
    
    if closest_idx < len(names):
        return names[closest_idx]
    return ""

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
if "ui_sens" not in st.session_state: st.session_state.ui_sens = 100

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
internal_threshold = 0.85 - (100 / 100.0) * 0.40

st.write(st.session_state.ui_sens)
st.write(internal_threshold)

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

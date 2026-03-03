import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import math
from pdf2image import convert_from_bytes
from streamlit_image_coordinates import streamlit_image_coordinates

# ==========================================
# 1. 楽譜解析エンジン (V8 修正版: 太線除外ロジック追加)
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

    # --- [追加] 五線より太い線（連桁など）を抽出する工程 ---
    # 五線の太さを推定 (通常 staff_space の 1/10 程度)
    line_thickness = max(1, int(staff_space * 0.12))
    # 五線の3倍以上の太さがあるものを抽出するためのカーネル
    thick_k_size = int(line_thickness * 3)
    thick_k = cv2.getStructuringElement(cv2.MORPH_RECT, (thick_k_size, thick_k_size))
    # オープニング処理で細い線（五線）を消し、太い部分だけ残す
    thick_elements = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, thick_k)
    
    # 符幹を切断
    sever_k_size = max(2, int(staff_space * 0.35))
    sever_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sever_k_size, sever_k_size))
    severed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, sever_k)

    num_labels, labels_sev, stats_sev, _ = cv2.connectedComponentsWithStats(severed, connectivity=8)
    notes_only = np.zeros_like(thresh)

    for label in range(1, num_labels):
        x, y, w, h, area = stats_sev[label]
        
        # --- 太い線と重なっている成分を除外するロジック ---
        roi_thick = thick_elements[y:y+h, x:x+w]
        # その成分の領域内に、太い線判定されたピクセルが一定以上あれば除外
        if np.any(roi_thick[labels_sev[y:y+h, x:x+w] == label]):
             continue

        if w > staff_space * 2.0 or h > staff_space * 4.0 or area > staff_space**2 * 3.0:
            continue
        if area < staff_space**2 * 0.4:
            continue

        notes_only[labels_sev == label] = 255

    # クロージング
    close_k_size = max(2, int(staff_space * 0.4))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k_size, close_k_size))
    filled = cv2.morphologyEx(notes_only, cv2.MORPH_CLOSE, close_k)

    # テンプレートマッチング
    nw, nh = int(staff_space * 1.3), int(staff_space * 1.0)
    template = np.zeros((nh, nw), dtype=np.uint8)
    cv2.ellipse(template, (nw // 2, nh // 2), (nw // 2 - 1, nh // 2 - 1), -20, 0, 360, 255, -1)
    res = cv2.matchTemplate(filled, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold_val)

    staff_centers = [np.mean(s) for s in staves]
    raw_rects, raw_scores = [], []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)

    for (x, y) in zip(*loc[::-1]):
        w, h = nw, nh
        score = res[y, x]
        cx, cy = x + w // 2, y + h // 2
        dist_to_nearest_staff = min(abs(cy - c) for c in staff_centers)
        if dist_to_nearest_staff > staff_space * 3.0 or cy >= labels.shape[0] or cx >= labels.shape[1]:
            continue
        
        label_id = labels[cy, cx]
        if label_id == 0: continue

        comp_w, comp_h, comp_area = stats[label_id, cv2.CC_STAT_WIDTH], stats[label_id, cv2.CC_STAT_HEIGHT], stats[label_id, cv2.CC_STAT_AREA]
        if comp_w > staff_space * 3.0 or comp_h > staff_space * 5.0 or comp_area > staff_space**2 * 2.5 or comp_area < staff_space**2 * 0.4:
            continue

        patch = filled[y:y + h, x:x + w]
        contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < (staff_space**2) * 0.4: continue

        rect = cv2.minAreaRect(cnt)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0: continue
        rotated_aspect = max(rw, rh) / min(rw, rh)
        if rotated_aspect < 1.1 or rotated_aspect > 2.0 or area / (rw * rh) > 0.85:
            continue

        raw_rects.append([x, y, x + w, y + h])
        raw_scores.append(score)

    nms_boxes = nms_v8_strict(np.array(raw_rects), np.array(raw_scores), staff_space) if raw_rects else []
    
    # 幹チェック
    final_boxes = []
    stem_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(staff_space * 0.6))))
    for box in nms_boxes:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        w = x2 - x1
        sw = int(w * 0.25)
        # 上下の幹を確認
        up_roi = thresh[max(0, y1-int(staff_space*2)):y1, max(0, cx-sw//2):min(thresh.shape[1], cx+sw//2)]
        down_roi = thresh[y2:min(thresh.shape[0], y2+int(staff_space*2)), max(0, cx-sw//2):min(thresh.shape[1], cx+sw//2)]
        if (up_roi.size > 0 and np.sum(cv2.morphologyEx(up_roi, cv2.MORPH_OPEN, stem_k)) > 0) or \
           (down_roi.size > 0 and np.sum(cv2.morphologyEx(down_roi, cv2.MORPH_OPEN, stem_k)) > 0):
            final_boxes.append(box)

    return np.array(final_boxes) if final_boxes else [], thick_elements

# ==========================================
# 2. 描画・キャッシュ処理 (デバッグ表示追加)
# ==========================================
def draw_all_notes(pil_img, auto_notes, custom_clicks, deleted_auto, staves, space, custom_labels, thick_elements=None, hide_boxes=False, selected_pos=None, erase_start=None):
    result = pil_img.copy().convert("RGB")
    
    # --- [デバッグ用] 太い線を青色でオーバーレイ表示 ---
    if thick_elements is not None and not hide_boxes:
        thick_overlay = np.zeros((result.height, result.width, 3), dtype=np.uint8)
        thick_overlay[thick_elements > 0] = [0, 100, 255] # 濃い水色
        res_array = np.array(result)
        # 透過で重ね合わせ
        mask = thick_elements > 0
        res_array[mask] = cv2.addWeighted(res_array[mask], 0.5, thick_overlay[mask], 0.5, 0)
        result = Image.fromarray(res_array)

    draw = ImageDraw.Draw(result)
    font_size = max(15, int(space))
    try: font = ImageFont.truetype("NotoSansJP-Regular.ttf", font_size)
    except: font = ImageFont.load_default()

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
    
    # グルーピングとラベル描画 (既存ロジック)
    groups = []
    for note in active_notes:
        if not groups: groups.append([note])
        else:
            last = groups[-1]
            if abs(note["x"] - (sum(n["x"] for n in last)/len(last))) < space * 0.8 and note["s_idx"] == last[0]["s_idx"]:
                last.append(note)
            else: groups.append([note])

    for group in groups:
        group.sort(key=lambda n: n["y"])
        labels_to_draw = []
        for note in group:
            x, y, s_idx, is_custom = note["x"], note["y"], note["s_idx"], note["is_custom"]
            p_name = next((l for (lx, ly), l in custom_labels.items() if math.hypot(x-lx, y-ly) < 2.0), None)
            if p_name is None:
                p_name = get_pitch_name(y, staves[s_idx], "treble" if s_idx % 2 == 0 else "bass")
            if p_name: labels_to_draw.append(p_name)
            if not hide_boxes:
                color = (255, 165, 0) if is_custom else (0, 255, 0)
                if selected_pos and math.hypot(x - selected_pos[0], y - selected_pos[1]) < 2:
                    draw.rectangle([x-int(space*0.6), y-int(space*0.5), x+int(space*0.6), y+int(space*0.5)], outline=(0,0,255), width=4)
                else:
                    draw.rectangle([x-int(space*0.6), y-int(space*0.5), x+int(space*0.6), y+int(space*0.5)], outline=color, width=2)

        if labels_to_draw:
            combined_text = " ".join(labels_to_draw)
            tx, ty = group[0]["x"] - int(space*0.6), group[0]["y"] - int(space*1.6)
            draw.text((tx, ty), combined_text, font=font, fill=(255, 0, 0))

    if erase_start and not hide_boxes:
        ex, ey = erase_start
        draw.ellipse([ex-4, ey-4, ex+4, ey+4], fill=(255,0,0))
        draw.line((ex, 0, ex, result.height), fill=(255,0,0), width=1)
        draw.line((0, ey, result.width, ey), fill=(255,0,0), width=1)
    return result

# get_pitch_name 関数は変更なしのため省略 (お手元のものをそのままお使いください)
def get_pitch_name(note_y, staff, clef):
    line1, line5 = staff[0], staff[4]
    step_size = abs(line1 - line5) / 8.0
    steps = int(round((line1 - note_y) / step_size))
    if clef == "treble":
        mapping = {-4:"ラ",-3:"シ",-2:"ド",-1:"レ",0:"ミ",1:"ファ",2:"ソ",3:"ラ",4:"シ",5:"ド",6:"レ",7:"ミ",8:"ファ",9:"ソ",10:"ラ",11:"シ",12:"ド",13:"レ",14:"ミ",15:"ファ",16:"ソ"}
    else:
        mapping = {-6:"ラ",-5:"シ",-4:"ド",-3:"レ",-2:"ミ",-1:"ファ",0:"ソ",1:"ラ",2:"シ",3:"ド",4:"レ",5:"ミ",6:"ファ",7:"ソ",8:"ラ",9:"シ",10:"ド",11:"レ",12:"ミ",13:"ファ",14:"ソ",15:"ラ",16:"シ"}
    return mapping.get(steps, "")

@st.cache_data(show_spinner=False)
def process_pdf_and_detect(pdf_bytes, internal_threshold):
    imgs = convert_from_bytes(pdf_bytes)
    data = []
    for img in imgs:
        staves, space = detect_staff_groups_v8(img)
        if staves:
            notes, thick = detect_note_heads_v8(np.array(img.convert('L')), space, internal_threshold, staves)
        else:
            notes, thick = [], None
        data.append({"image": img, "staves": staves, "space": space, "notes": notes, "thick": thick})
    return data

# ==========================================
# 3. Streamlit UI (Step 2に thick を渡すよう修正)
# ==========================================
# ... (中略: st.session_state などの初期化部分は変更なし) ...
st.set_page_config(page_title="ドレミ付与 V8", layout="wide") 

if "step" not in st.session_state: st.session_state.step = 1
if "custom_clicks" not in st.session_state: st.session_state.custom_clicks = {}
if "deleted_auto_notes" not in st.session_state: st.session_state.deleted_auto_notes = {}
if "custom_labels" not in st.session_state: st.session_state.custom_labels = {} 
if "selected_note" not in st.session_state: st.session_state.selected_note = None 
if "pdf_data" not in st.session_state: st.session_state.pdf_data = None

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1

st.title("🎼 ドレミ付与ツール V8 (太線除外Ver.)")
# ... (Step 表示部分は変更なし) ...

FIXED_DISP_WIDTH = 800 
internal_threshold = 0.85 - (100 / 100.0) * 0.40

if st.session_state.pdf_data:
    pages = process_pdf_and_detect(st.session_state.pdf_data, internal_threshold)
else:
    pages = []

if st.session_state.step == 1:
    st.subheader("Step 1: 楽譜PDFをアップロード")
    up = st.file_uploader("PDFファイルを選択してください", type="pdf")
    if up:
        st.session_state.pdf_data = up.getvalue()
        st.button("次へ ➡️", on_click=next_step, type="primary")

if st.session_state.step == 2:
    subheader_col1, subheader_col2 = st.columns([2, 1])
    subheader_col1.subheader("Step 2: 自動検出の調整")
    st.info("💡 青色で塗られている部分は「太い線（連桁など）」として検知され、音符から除外されています。")
    subheader_col2.button("次へ：テキストの微調整 ➡️", on_click=next_step, type="primary")

    img_container_col, slider_container_col = st.columns([4, 1]) 
    with slider_container_col:
        edit_mode = st.radio("画像クリック時の動作", ["👆 通常", "🔲 範囲消去"])

    with img_container_col:
        for i, page in enumerate(pages):
            st.write(f"### ページ {i + 1}")
            if page["staves"]:
                clicks = st.session_state.custom_clicks.setdefault(i, [])
                deleted_auto = st.session_state.deleted_auto_notes.setdefault(i, [])
                erase_start_key = f"s2_erase_start_{i}"
                if erase_start_key not in st.session_state: st.session_state[erase_start_key] = None

                res_img = draw_all_notes(
                    page["image"], page["notes"], clicks, deleted_auto, page["staves"], page["space"], 
                    st.session_state.custom_labels.get(i, {}), thick_elements=page.get("thick"),
                    erase_start=st.session_state[erase_start_key]
                )
                
                value = streamlit_image_coordinates(res_img, key=f"s2_img_{i}", width=FIXED_DISP_WIDTH)
                
                if value:
                    scale = page["image"].width / FIXED_DISP_WIDTH
                    real_x, real_y = value["x"] * scale, value["y"] * scale
                    # --- クリック処理 (省略: 提示コードと同じ) ---
                    # 簡略化のためここでの st.rerun() 等の処理はお手元のものを維持してください
                    pass 

# Step 3, 4 も同様に pages[i].get("thick") を渡さない、または None を渡すことでデバッグ表示を消せます
# ... (以下略) ...

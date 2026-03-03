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
    
    # 閾値以上の位置を特定
    loc = np.where(res >= threshold_val)

    staff_centers = [np.mean(s) for s in staves]
    raw_rects, raw_scores = [], []
    
    # 隔離チェック用のマージン
    chk_w = int(staff_space * 2.5)
    chk_h = int(staff_space * 3.5)
    img_h, img_w = filled.shape

    for (x, y) in zip(*loc[::-1]):
        w, h = nw, nh
        score = res[y, x]
        cx, cy = x + w//2, y + h//2
        dist_to_nearest_staff = min(abs(cy - c) for c in staff_centers)
        if dist_to_nearest_staff > staff_space * 4.0: continue
        
        # 1. 広い窓での隔離チェック（連桁/長い線の排除）
        chk_x1, chk_x2 = max(0, cx - chk_w // 2), min(img_w, cx + chk_w // 2)
        chk_y1, chk_y2 = max(0, cy - chk_h // 2), min(img_h, cy + chk_h // 2)
        
        patch_check = filled[chk_y1:chk_y2, chk_x1:chk_x2]
        contours, _ = cv2.findContours(patch_check, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        
        pcx, pcy = cx - chk_x1, cy - chk_y1
        best_cnt = None
        min_dist = float('inf')
        for c_ in contours:
            M = cv2.moments(c_)
            if M["m00"] == 0: continue
            dist = math.hypot(int(M["m10"]/M["m00"]) - pcx, int(M["m01"]/M["m00"]) - pcy)
            if dist < min_dist:
                min_dist = dist
                best_cnt = c_
                
        if best_cnt is None or min_dist > staff_space * 0.8: continue
        
        # 窓の端に触れていたらノイズ
        ph, pw = patch_check.shape
        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        if bx <= 1 or by <= 1 or (bx + bw) >= pw - 1 or (by + bh) >= ph - 1:
            continue
            
        # 2. 形状チェック
        area = cv2.contourArea(best_cnt)
        if area < (staff_space**2) * 0.3: continue
        
        rect = cv2.minAreaRect(best_cnt)
        (rw, rh) = rect[1]
        if rw == 0 or rh == 0: continue
        if max(rw, rh) / min(rw, rh) > 2.2: continue 
        if area / (rw * rh) > 0.88: continue 
        
        peri = cv2.arcLength(best_cnt, True)
        if peri > 0 and (4.0 * np.pi * area / (peri**2)) >= 0.55:
            hull = cv2.convexHull(best_cnt)
            if cv2.contourArea(hull) > 0 and (area / cv2.contourArea(hull)) >= 0.85:
                raw_rects.append([cx - w//2, cy - h//2, cx + w//2, cy + h//2])
                raw_scores.append(score)
    
    nms_boxes = nms_v8_strict(np.array(raw_rects), np.array(raw_scores), staff_space) if raw_rects else []
    if not nms_boxes.any(): return []

    # 3. 符幹チェック
    final_boxes = []
    stem_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(staff_space * 0.5))))
    for box in nms_boxes:
        x1, y1, x2, y2 = box
        cx, w = (x1 + x2) // 2, x2 - x1
        has_stem = False
        for (sy1, sy2) in [(max(0, y1-int(staff_space*1.5)), y1), (y2, min(thresh.shape[0], y2+int(staff_space*1.5)))]:
            sx1, sx2 = max(0, cx-int(w*0.15)), min(thresh.shape[1], cx+int(w*0.15))
            if sy2 > sy1 and sx2 > sx1:
                if np.sum(cv2.morphologyEx(thresh[sy1:sy2, sx1:sx2], cv2.MORPH_OPEN, stem_k)) > 0:
                    has_stem = True; break
        if has_stem: final_boxes.append(box)

    return np.array(final_boxes) if final_boxes else []

def get_pitch_name(note_y, staff, clef):
    line1, line5 = staff[0], staff[4]
    step_size = abs(line1 - line5) / 8.0
    steps = int(round((line1 - note_y) / step_size))
    mapping = {"treble": {-4:"ラ",-3:"シ",-2:"ド",-1:"レ",0:"ミ",1:"ファ",2:"ソ",3:"ラ",4:"シ",5:"ド",6:"レ",7:"ミ",8:"ファ",9:"ソ",10:"ラ",11:"シ",12:"ド",13:"レ",14:"ミ",15:"ファ",16:"ソ"},
               "bass": {-6:"ド",-5:"レ",-4:"ミ",-3:"ファ",-2:"ソ",-1:"ラ",0:"シ",1:"ド",2:"レ",3:"ミ",4:"ファ",5:"ソ",6:"ラ",7:"シ",8:"ド",9:"レ",10:"ミ",11:"ファ",12:"ソ",13:"ラ",14:"シ"}}
    return mapping[clef].get(steps, "")

# ==========================================
# 2. 描画・キャッシュ処理 
# ==========================================
def draw_all_notes(pil_img, auto_notes, custom_clicks, deleted_auto, staves, space, custom_labels, hide_boxes=False, selected_pos=None, erase_start=None):
    result = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    font_size = max(15, int(space))
    try: font = ImageFont.truetype("NotoSansJP-Regular.ttf", font_size)
    except: font = ImageFont.load_default()

    drawn_text_rects, active_notes = [], []
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

    for group in groups:
        group.sort(key=lambda n: n["y"]) 
        labels = []
        for note in group:
            p_name = next((l for (lx, ly), l in custom_labels.items() if math.hypot(note["x"]-lx, note["y"]-ly) < 2.0), None)
            if p_name is None: p_name = get_pitch_name(note["y"], staves[note["s_idx"]], "treble" if note["s_idx"] % 2 == 0 else "bass")
            if p_name: labels.append(p_name)
            if not hide_boxes:
                hw, hh = int(space * 0.6), int(space * 0.5)
                color = (0,0,255) if (selected_pos and math.hypot(note["x"]-selected_pos[0], note["y"]-selected_pos[1]) < 2) else ((255,165,0) if note["is_custom"] else (0,255,0))
                draw.rectangle([note["x"]-hw, note["y"]-hh, note["x"]+hw, note["y"]+hh], outline=color, width=2 if color != (0,0,255) else 4)

        if labels and (not hide_boxes or any(labels)):
            combined = " ".join(labels)
            tx, ty = group[0]["x"] - int(space*0.6), group[0]["y"] - int(space*1.6)
            tw, th = len(combined)*font_size, font_size
            while any(not (tx + tw < rx or tx > rx + rw or ty + th < ry or ty > ry + rh) for (rx,ry,rw,rh) in drawn_text_rects):
                tx += int(font_size * 1.1)
            draw.text((tx, ty), combined, font=font, fill=(255, 0, 0))
            drawn_text_rects.append((tx, ty, tw, th))

    if erase_start and not hide_boxes:
        ex, ey = erase_start
        r = max(4, int(space * 0.4))
        draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(255, 0, 0))
        draw.line((ex, 0, ex, result.height), fill=(255, 0, 0), width=1); draw.line((0, ey, result.width, ey), fill=(255, 0, 0), width=1)
    return result

@st.cache_data(show_spinner=False)
def process_pdf_and_detect(pdf_bytes, threshold):
    imgs = convert_from_bytes(pdf_bytes)
    data = []
    for img in imgs:
        staves, space = detect_staff_groups_v8(img)
        notes = detect_note_heads_v8(np.array(img.convert('L')), space, threshold, staves) if staves else []
        data.append({"image": img, "staves": staves, "space": space, "notes": notes})
    return data

# ==========================================
# 3. Streamlit UI
# ==========================================
st.set_page_config(page_title="ドレミ付与 V8", layout="wide") 
for k in ["step", "custom_clicks", "deleted_auto_notes", "custom_labels", "selected_note", "pdf_data"]:
    if k not in st.session_state: st.session_state[k] = (1 if k=="step" else (None if k in ["selected_note","pdf_data"] else {}))

def next_step(): st.session_state.step += 1
def prev_step(): st.session_state.step -= 1

st.title("🎼 ドレミ付与ツール V8")
steps = ["1. アップロード", "2. ワンクリック調整", "3. マニュアル微調整", "4. プレビュー＆保存"]
cols = st.columns(4)
for i, s in enumerate(steps): cols[i].markdown(f"**🔵 {s}**" if st.session_state.step == i+1 else f"⚪ {s}")
st.divider()

# 🛑 修正箇所: 感度100に対応する低閾値 (0.40) に固定
FIXED_THRESHOLD = 0.40
FIXED_WIDTH = 800

if st.session_state.pdf_data: pages = process_pdf_and_detect(st.session_state.pdf_data, FIXED_THRESHOLD)
else: pages = []

if st.session_state.step == 1:
    st.subheader("Step 1: 楽譜PDFをアップロード")
    up = st.file_uploader("PDFファイルを選択してください", type="pdf")
    if up:
        st.session_state.pdf_data = up.getvalue()
        st.success("読込完了！"); st.button("次へ ➡️", on_click=next_step, type="primary")

if st.session_state.step == 2:
    h1, h2 = st.columns([2, 1])
    h1.subheader("Step 2: 自動検出の調整")
    h2.button("次へ：テキストの微調整 ➡️", on_click=next_step, type="primary"); h2.button("⬅️ やり直す", on_click=prev_step)
    c1, c2 = st.columns([4, 1])
    with c2:
        st.write("### "); st.subheader("🖱️ 操作ガイド"); st.info("💡 検出感度は最高(100)固定です。")
        mode = st.radio("動作", ["👆 通常", "🔲 範囲消去"])
    with c1:
        for i, page in enumerate(pages):
            st.write(f"### ページ {i + 1}")
            if page["staves"]:
                clicks, deleted = st.session_state.custom_clicks.setdefault(i, []), st.session_state.deleted_auto_notes.setdefault(i, [])
                es_key = f"s2_es_{i}"
                if es_key not in st.session_state: st.session_state[es_key] = None
                if "範囲消去" not in mode: st.session_state[es_key] = None
                res = draw_all_notes(page["image"], page["notes"], clicks, deleted, page["staves"], page["space"], st.session_state.custom_labels.get(i, {}), erase_start=st.session_state[es_key])
                val = streamlit_image_coordinates(res, key=f"s2_img_{i}", width=FIXED_WIDTH)
                lc_key = f"s2_lc_{i}"
                if val and val != st.session_state.get(lc_key):
                    st.session_state[lc_key] = val
                    rx, ry = val["x"] * (page["image"].width / FIXED_WIDTH), val["y"] * (page["image"].width / FIXED_WIDTH)
                    if "範囲消去" in mode:
                        if st.session_state[es_key] is None: st.session_state[es_key] = (rx, ry)
                        else:
                            x1, y1 = st.session_state[es_key]; x2, y2 = rx, ry
                            clicks[:] = [p for p in clicks if not (min(x1,x2) <= p[0] <= max(x1,x2) and min(y1,y2) <= p[1] <= max(y1,y2))]
                            for b in page["notes"]:
                                cx, cy = (b[0]+b[2])//2, (b[1]+b[3])//2
                                if min(x1,x2) <= cx <= max(x1,x2) and min(y1,y2) <= cy <= max(y1,y2):
                                    if not any(math.hypot(cx-dx,cy-dy) < 2.0 for dx,dy in deleted): deleted.append((cx,cy))
                            st.session_state[es_key] = None
                    else:
                        hit_x, hit_y = page["space"]*0.8, page["space"]*0.45
                        done = False
                        for p in clicks.copy():
                            if abs(rx-p[0]) < hit_x and abs(ry-p[1]) < hit_y: clicks.remove(p); done=True; break
                        if not done:
                            for b in page["notes"]:
                                cx, cy = (b[0]+b[2])//2, (b[1]+b[3])//2
                                if not any(math.hypot(cx-dx,cy-dy) < 2.0 for dx,dy in deleted):
                                    if abs(rx-cx) < hit_x and abs(ry-cy) < hit_y: deleted.append((cx,cy)); done=True; break
                        if not done: clicks.append((rx, ry))
                    st.rerun()
            else: st.warning(f"⚠️ ページ {i+1} 五線検出不能"); st.image(page["image"], width=FIXED_WIDTH)

if st.session_state.step == 3:
    col1, col2 = st.columns([1, 1]); col1.subheader("Step 3: マニュアル微調整")
    col2.button("次へ：完成プレビュー ➡️", on_click=next_step, type="primary"); col2.button("⬅️ 戻る", on_click=prev_step)
    for i, p in enumerate(pages):
        st.write(f"### ページ {i+1}")
        if p["staves"]:
            clicks, deleted = st.session_state.custom_clicks.get(i, []), st.session_state.deleted_auto_notes.get(i, [])
            labels = st.session_state.custom_labels.setdefault(i, {})
            sel_p = (st.session_state.selected_note["x"], st.session_state.selected_note["y"]) if st.session_state.selected_note and st.session_state.selected_note["page"] == i else None
            res = draw_all_notes(p["image"], p["notes"], clicks, deleted, p["staves"], p["space"], labels, selected_pos=sel_p)
            val = streamlit_image_coordinates(res, key=f"s3_img_{i}", width=FIXED_WIDTH)
            lc_key = f"s3_lc_{i}"
            if val and val != st.session_state.get(lc_key):
                st.session_state[lc_key] = val
                rx, ry = val["x"] * (p["image"].width / FIXED_WIDTH), val["y"] * (p["image"].width / FIXED_WIDTH)
                hit_x, hit_y = p["space"]*0.8, p["space"]*0.45
                found = next((pt for pt in clicks if abs(rx-pt[0]) < hit_x and abs(ry-pt[1]) < hit_y), None)
                if not found:
                    found_box = next((b for b in p["notes"] if not any(math.hypot((b[0]+b[2])/2-dx,(b[1]+b[3])/2-dy) < 2.0 for dx,dy in deleted) and abs(rx-(b[0]+b[2])/2) < hit_x and abs(ry-(b[1]+b[3])/2) < hit_y), None)
                    if found_box: found = ((found_box[0]+found_box[2])//2, (found_box[1]+found_box[3])//2)
                if not found: st.session_state.custom_clicks[i].append((rx,ry)); found=(int(rx),int(ry))
                cur_l = labels.get(found)
                if cur_l is None:
                    idx = np.argmin([abs(np.mean(s)-found[1]) for s in p["staves"]])
                    cur_l = get_pitch_name(found[1], p["staves"][idx], "treble" if idx % 2 == 0 else "bass")
                st.session_state.selected_note = {"page":i, "x":found[0], "y":found[1], "label":cur_l}; st.rerun()
            if st.session_state.selected_note and st.session_state.selected_note["page"] == i:
                sel = st.session_state.selected_note
                new_l = st.text_input(f"✎ ページ {i+1} テキスト変更", value=sel["label"], key=f"in_{i}")
                if new_l != sel["label"]: labels[(sel["x"], sel["y"])] = new_l; st.session_state.selected_note["label"] = new_l; st.rerun()
        else: st.image(p["image"], width=FIXED_WIDTH)

if st.session_state.step == 4:
    st.subheader("Step 4: 完成プレビュー"); c1, c2 = st.columns([1, 1]); c2.button("⬅️ 戻る", on_click=prev_step)
    out = []
    for i, p in enumerate(pages):
        if p["staves"]:
            res = draw_all_notes(p["image"], p["notes"], st.session_state.custom_clicks.get(i,[]), st.session_state.deleted_auto_notes.get(i,[]), p["staves"], p["space"], st.session_state.custom_labels.get(i,{}), hide_boxes=True)
            out.append(res); st.image(res, width=FIXED_WIDTH)
        else: out.append(p["image"]); st.image(p["image"], width=FIXED_WIDTH)
    if out:
        buf = io.BytesIO(); out[0].save(buf, format="PDF", save_all=True, append_images=out[1:]); st.download_button("📥 楽譜PDFをダウンロード", data=buf.getvalue(), file_name="score.pdf", mime="application/pdf", type="primary", use_container_width=True)

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from pdf2image import convert_from_bytes

# ==========================================
# 1. 楽譜解析エンジン（V6：重複除去・和音強化版）
# ==========================================

def detect_staff_groups_precise(pil_img):
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

def nms_improved_for_chords(boxes, overlapThresh=0.4):
    """
    V6: 重複検知を強力に防ぎつつ、和音を残す改良型NMS
    """
    if len(boxes) == 0: return []
    boxes = boxes.astype("float")
    pick = []
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # 面積と、y座標の底（下端）でソート
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

        # 重なり割合 (Intersection over Union)
        overlap = (w * h) / area[idxs[:last]]
        
        # 横方向の重なり具合
        x_width = (x2[i] - x1[i] + 1)
        x_overlap = w / x_width
        
        # 【判定ロジック】
        # 1. 全体的に半分以上重なっているなら、同一音符とみなして削除
        # 2. 横位置がほぼ同じ(80%以上)で、かつ少しでも縦に重なっているなら、同一音符とみなして削除
        #    （ただし、和音として別個に認識させるため、縦の重なりが小さい場合は残す）
        delete_condition = (overlap > 0.5) | ((x_overlap > 0.8) & (overlap > 0.2))
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(delete_condition)[0])))
    
    return boxes[pick].astype("int")

def detect_note_heads_v6(gray_img, staff_space, user_threshold):
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # ノイズ除去
    k_size = max(2, int(staff_space * 0.3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    nw, nh = int(staff_space * 1.2), int(staff_space * 0.9)
    
    # テンプレート作成
    temp_filled = np.zeros((nh + 4, nw + 4), dtype=np.uint8)
    cv2.ellipse(temp_filled, (nw//2+2, nh//2+2), (nw//2, nh//2), -20, 0, 360, 255, -1)
    
    temp_hollow = np.zeros((nh + 4, nw + 4), dtype=np.uint8)
    thickness = max(1, int(staff_space * 0.2))
    cv2.ellipse(temp_hollow, (nw//2+2, nh//2+2), (nw//2, nh//2), -20, 0, 360, 255, thickness)

    res_f = cv2.matchTemplate(clean_thresh, temp_filled, cv2.TM_CCOEFF_NORMED)
    res_h = cv2.matchTemplate(clean_thresh, temp_hollow, cv2.TM_CCOEFF_NORMED)
    res = np.maximum(res_f, res_h)
    
    loc = np.where(res >= user_threshold)
    rects = [[int(pt[0]), int(pt[1]), int(pt[0]+nw), int(pt[1]+nh)] for pt in zip(*loc[::-1])]
    
    if not rects: return []
    # 改良版NMSを適用
    return nms_improved_for_chords(np.array(rects))

def get_pitch_name(note_y, staff, clef, flats_count):
    line1 = staff[0]
    line5 = staff[4]
    step_size = abs(line1 - line5) / 8.0
    diff = line1 - note_y
    steps = int(round(diff / step_size))

    if clef == "treble":
        mapping = {
            -4:"ラ", -3:"シ", -2:"ド", -1:"レ",
            0:"ミ", 1:"ファ", 2:"ソ", 3:"ラ", 4:"シ", 5:"ド", 6:"レ", 7:"ミ", 8:"ファ",
            9:"ソ", 10:"ラ", 11:"シ", 12:"ド", 13:"レ", 14:"ミ", 15:"ファ", 16:"ソ"
        }
    else: # bass
        mapping = {
            -6:"ド", -5:"レ", -4:"ミ", -3:"ファ", -2:"ソ", -1:"ラ",
            0:"シ", 1:"ド", 2:"レ", 3:"ミ", 4:"ファ", 5:"ソ", 6:"ラ", 7:"シ", 8:"ド",
            9:"レ", 10:"ミ", 11:"ファ", 12:"ソ", 13:"ラ", 14:"シ"
        }

    # ヘ音記号の基準微調整（画像によってズレる場合があるためマッピングを再確認）
    # 通常ヘ音記号の第1線は「ソ」ですが、検出結果に合わせて調整が必要な場合があります。
    
    name = mapping.get(steps, "")
    flat_order = ["シ", "ミ", "ラ", "レ", "ソ", "ド", "ファ"]
    is_flat = name in flat_order[:flats_count]
    return name, is_flat

def process_page_v6(pil_img, threshold, flats_count):
    staves, space = detect_staff_groups_precise(pil_img)
    if not staves: return pil_img

    img_gray = np.array(pil_img.convert('L'))
    notes = detect_note_heads_v6(img_gray, space, threshold)
    
    result = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    
    try: font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(15, int(space)))
    except: font = ImageFont.load_default()

    for box in notes:
        b_list = box.tolist()
        yc = (b_list[1] + b_list[3]) // 2
        s_idx = np.argmin([abs(np.mean(s) - yc) for s in staves])
        clef = "treble" if s_idx % 2 == 0 else "bass"
        
        p_name, is_flat = get_pitch_name(yc, staves[s_idx], clef, flats_count)
        
        if p_name:
            draw.rectangle(b_list, outline=(0, 255, 0), width=2)
            color = (0, 0, 255) if is_flat else (255, 0, 0)
            draw.text((b_list[0], b_list[1] - int(space * 1.6)), p_name, font=font, fill=color)

    return result

# Streamlit UI
st.set_page_config(page_title="ドレミ付与 V6", layout="centered")
st.title("🎼 ドレミ付与ツール V6")
st.write("二重検知を防止し、和音と白抜き音符の認識を最適化しました。")

st.sidebar.header("⚙️ 設定")
flats = st.sidebar.selectbox("調号（♭）の数", range(8), index=4) 
sens = st.sidebar.slider("検出感度", 0.3, 0.9, 0.5, 0.01)

up = st.file_uploader("PDFをアップロード", type="pdf")

if up:
    imgs = convert_from_bytes(up.read())
    with st.spinner('解析中...'):
        processed_imgs = [process_page_v6(im, sens, flats) for im in imgs]
        for im in processed_imgs:
            st.image(im, use_column_width=True)

    if processed_imgs:
        buf = io.BytesIO()
        processed_imgs[0].save(buf, format='PDF', save_all=True, append_images=processed_imgs[1:])
        st.download_button("完成版PDFをダウンロード", buf.getvalue(), "score_v6.pdf", "application/pdf")

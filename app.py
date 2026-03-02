import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from pdf2image import convert_from_bytes

# ==========================================
# 1. 楽譜解析エンジン（V5：白抜き・和音対応版）
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

def nms_for_chords(boxes, overlapThresh=0.2):
    """
    和音に対応した重複除去。
    上下方向（y軸）の重なりには寛容に、
    水平方向（x軸）が大きく重なっている場合のみ除去する。
    """
    if len(boxes) == 0: return []
    boxes = boxes.astype("float")
    pick = []
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
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

        # 水平方向の重なり具合を重視
        overlap = (w * h) / area[idxs[:last]]
        
        # 和音の場合、x座標がほぼ同じでyが少しずれるため、
        # overlapの閾値を厳しめにするか、xの重なりを個別チェック
        x_overlap = w / (x2[i] - x1[i] + 1)
        
        # x方向が8割以上重なり、かつ全体面積も一定以上重なる場合のみ削除
        delete_condition = (overlap > overlapThresh) & (x_overlap > 0.8)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(delete_condition)[0])))
    
    return boxes[pick].astype("int")

def detect_note_heads_v5(gray_img, staff_space, user_threshold):
    """
    塗りつぶし音符と白抜き音符の両方を検出
    """
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 五線除去のための処理
    k_size = max(2, int(staff_space * 0.3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    nw, nh = int(staff_space * 1.2), int(staff_space * 0.9)
    
    # テンプレート1: 塗りつぶし音符（4分音符など）
    temp_filled = np.zeros((nh + 4, nw + 4), dtype=np.uint8)
    cv2.ellipse(temp_filled, (nw//2+2, nh//2+2), (nw//2, nh//2), -20, 0, 360, 255, -1)
    
    # テンプレート2: 白抜き音符（2分音符など）
    temp_hollow = np.zeros((nh + 4, nw + 4), dtype=np.uint8)
    thickness = max(1, int(staff_space * 0.2))
    cv2.ellipse(temp_hollow, (nw//2+2, nh//2+2), (nw//2, nh//2), -20, 0, 360, 255, thickness)

    res_f = cv2.matchTemplate(clean_thresh, temp_filled, cv2.TM_CCOEFF_NORMED)
    res_h = cv2.matchTemplate(clean_thresh, temp_hollow, cv2.TM_CCOEFF_NORMED)
    
    # 両方の結果を統合
    res = np.maximum(res_f, res_h)
    
    loc = np.where(res >= user_threshold)
    rects = [[int(pt[0]), int(pt[1]), int(pt[0]+nw), int(pt[1]+nh)] for pt in zip(*loc[::-1])]
    
    if not rects: return []
    return nms_for_chords(np.array(rects), overlapThresh=0.15)

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
            9:"ソ", 10:"ラ", 11:"シ", 12:"ド", 13:"レ", 14:"ミ"
        }
    else: # bass
        mapping = {
            -4:"ファ", -3:"ソ", -2:"ミ", -1:"ファ", # 下加線対応
            0:"ソ", 1:"ラ", 2:"シ", 3:"ド", 4:"レ", 5:"ミ", 6:"ファ", 7:"ソ", 8:"ラ",
            9:"シ", 10:"ド", 11:"レ", 12:"ミ", 13:"ファ", 14:"ソ"
        }

    name = mapping.get(steps, "")
    flat_order = ["シ", "ミ", "ラ", "レ", "ソ", "ド", "ファ"]
    is_flat = name in flat_order[:flats_count]
    return name, is_flat

def process_page_v5(pil_img, threshold, flats_count):
    staves, space = detect_staff_groups_precise(pil_img)
    if not staves: return pil_img

    img_gray = np.array(pil_img.convert('L'))
    notes = detect_note_heads_v5(img_gray, space, threshold)
    
    result = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    
    try: font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(15, int(space)))
    except: font = ImageFont.load_default()

    for box in notes:
        b_list = box.tolist()
        yc = (b_list[1] + b_list[3]) // 2
        
        # 最も近い五線を特定
        s_idx = np.argmin([abs(np.mean(s) - yc) for s in staves])
        clef = "treble" if s_idx % 2 == 0 else "bass"
        
        p_name, is_flat = get_pitch_name(yc, staves[s_idx], clef, flats_count)
        
        if p_name:
            draw.rectangle(b_list, outline=(0, 255, 0), width=2)
            color = (0, 0, 255) if is_flat else (255, 0, 0)
            draw.text((b_list[0], b_list[1] - int(space * 1.6)), p_name, font=font, fill=color)

    return result

# Streamlit UI
st.set_page_config(page_title="ドレミ付与 V5", layout="centered")
st.title("🎼 ドレミ付与ツール V5 (白抜き・和音対応)")
st.write("2分音符や、縦に並んだ和音の検出精度を強化しました。")

st.sidebar.header("⚙️ 設定")
flats = st.sidebar.selectbox("調号（♭）の数", range(8), index=4) 
sens = st.sidebar.slider("検出感度 (和音が漏れる場合は下げてみてください)", 0.3, 0.9, 0.55, 0.01)

up = st.file_uploader("PDFをアップロード", type="pdf")

if up:
    imgs = convert_from_bytes(up.read())
    with st.spinner('解析中...'):
        processed_imgs = [process_page_v5(im, sens, flats) for im in imgs]
        for im in processed_imgs:
            st.image(im, use_column_width=True)

    if processed_imgs:
        buf = io.BytesIO()
        processed_imgs[0].save(buf, format='PDF', save_all=True, append_images=processed_imgs[1:])
        st.download_button("完成版PDFをダウンロード", buf.getvalue(), "score_v5.pdf", "application/pdf")

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from pdf2image import convert_from_bytes

# ==========================================
# 1. 楽譜解析エンジン（高精度版）
# ==========================================

def detect_staff_groups_precise(pil_img):
    """
    水平投影を使用して、正確に5本1組の五線を検出する。
    """
    img_array = np.array(pil_img.convert('L'))
    # 二値化
    _, thresh = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 横方向の密度を計算
    projection = np.sum(thresh, axis=1)
    
    # 線の候補（ピーク）を抽出
    peaks = []
    thresh_val = np.max(projection) * 0.5
    for i in range(1, len(projection) - 1):
        if projection[i] > thresh_val and projection[i] >= projection[i-1] and projection[i] >= projection[i+1]:
            peaks.append(i)

    # 近すぎるピークを統合
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

    # 【重要】5本1組のバリデーション
    staves = []
    i = 0
    while i <= len(merged) - 5:
        segment = merged[i:i+5]
        diffs = np.diff(segment)
        avg_spacing = np.mean(diffs)
        
        # 5本の線の間隔が均等（誤差25%以内）かチェック
        if np.all(np.abs(diffs - avg_spacing) < avg_spacing * 0.25):
            # [第1線(下), 第2, 第3, 第4, 第5線(上)] の順に並び替え
            staves.append(sorted(segment, reverse=True))
            i += 5
        else:
            i += 1
            
    return staves, avg_spacing if staves else 10

def get_pitch_name(note_y, staff, clef, flats_count):
    """
    一番下の線（第1線）からの距離で音を判定
    """
    line1 = staff[0]  # 第1線（下）
    line5 = staff[4]  # 第5線（上）
    
    # 1ステップ（線と間の距離）の算出
    # 4つの間があるため、(line1-line5)/8 が1ステップ
    step_size = abs(line1 - line5) / 8.0
    
    # 第1線からの距離をステップ数に変換
    # note_yが小さいほど高い音なので (line1 - note_y)
    diff = line1 - note_y
    steps = int(round(diff / step_size))

    # 音階マッピング (0 = 第1線)
    if clef == "treble":
        # 第1線(0)はミ
        mapping = {
            -4:"ラ", -3:"シ", -2:"ド", -1:"レ",
            0:"ミ", 1:"ファ", 2:"ソ", 3:"ラ", 4:"シ", 5:"ド", 6:"レ", 7:"ミ", 8:"ファ",
            9:"ソ", 10:"ラ", 11:"シ", 12:"ド", 13:"レ", 14:"ミ"
        }
    else: # bass
        # 第1線(0)はソ
        mapping = {
            -2:"ミ", -1:"ファ",
            0:"ソ", 1:"ラ", 2:"シ", 3:"ド", 4:"レ", 5:"ミ", 6:"ファ", 7:"ソ", 8:"ラ",
            9:"シ", 10:"ド", 11:"レ", 12:"ミ"
        }

    name = mapping.get(steps, "")
    
    # ♭判定
    flat_order = ["シ", "ミ", "ラ", "レ", "ソ", "ド", "ファ"]
    is_flat = name in flat_order[:flats_count]
    
    return name, is_flat

# ==========================================
# 2. メイン処理・UI
# ==========================================

# ※ non_max_suppression_fast と detect_note_heads_precise は
# 以前のコードで動作していたため、そのまま統合します。

def non_max_suppression_fast(boxes, overlapThresh):
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
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return boxes[pick].astype("int")

def detect_note_heads_v3(gray_img, staff_space, user_threshold):
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k_size = max(2, int(staff_space * 0.5)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    nw, nh = int(staff_space * 1.2), int(staff_space * 0.9)
    template = np.zeros((nh + 4, nw + 4), dtype=np.uint8)
    cv2.ellipse(template, (nw//2+2, nh//2+2), (nw//2, nh//2), -20, 0, 360, 255, -1)
    res = cv2.matchTemplate(clean_thresh, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= user_threshold)
    rects = [[int(pt[0]), int(pt[1]), int(pt[0]+nw), int(pt[1]+nh)] for pt in zip(*loc[::-1])]
    if not rects: return []
    return non_max_suppression_fast(np.array(rects), overlapThresh=0.3)

def process_page(pil_img, threshold, flats_count):
    staves, space = detect_staff_groups_precise(pil_img)
    if not staves: return pil_img

    img_gray = np.array(pil_img.convert('L'))
    notes = detect_note_heads_v3(img_gray, space, threshold)
    
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
            # 文字が重ならないよう少し上に配置
            draw.text((b_list[0], b_list[1] - int(space * 1.6)), p_name, font=font, fill=color)

    return result

# Streamlit 設定
st.set_page_config(page_title="ドレミ付与ツール V4", layout="centered")
st.title("🎼 楽譜ドレミ付与ツール V4")
st.write("高精細な五線検出により、加線の音階ミスを修正しました。")

st.sidebar.header("⚙️ 設定")
flats = st.sidebar.selectbox("調号（♭）の数", range(8), index=4) 
sens = st.sidebar.slider("検出感度", 0.4, 0.95, 0.65, 0.01)

up = st.file_uploader("PDFをアップロードしてください", type="pdf")

if up:
    imgs = convert_from_bytes(up.read())
    with st.spinner('解析中...'):
        processed_imgs = [process_page(im, sens, flats) for im in imgs]
        for im in processed_imgs:
            st.image(im, use_column_width=True)

    if processed_imgs:
        buf = io.BytesIO()
        processed_imgs[0].save(buf, format='PDF', save_all=True, append_images=processed_imgs[1:])
        st.download_button("完成版PDFをダウンロード", buf.getvalue(), "score_doremi.pdf", "application/pdf")

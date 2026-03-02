import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
from pdf2image import convert_from_bytes

# ==========================================
# 1. 楽譜解析エンジン（V7：白抜き穴埋め・重心統合版）
# ==========================================

def detect_staff_groups_v7(pil_img):
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

def nms_v7_centroid(boxes, scores, staff_space):
    """
    重複除去V7: 中心点間の距離に基づいて、近すぎる検出を統合する
    """
    if len(boxes) == 0: return []
    
    # スコアが高い順に並び替え
    idxs = np.argsort(scores)[::-1]
    pick = []
    
    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)
        
        # 現在の音符の中心点
        curr_center_x = (boxes[i, 0] + boxes[i, 2]) / 2
        curr_center_y = (boxes[i, 1] + boxes[i, 3]) / 2
        
        remaining_idxs = idxs[1:]
        if len(remaining_idxs) == 0: break
        
        # 他の音符の中心点
        other_centers_x = (boxes[remaining_idxs, 0] + boxes[remaining_idxs, 2]) / 2
        other_centers_y = (boxes[remaining_idxs, 1] + boxes[remaining_idxs, 3]) / 2
        
        # 距離を計算
        dist = np.sqrt((curr_center_x - other_centers_x)**2 + (curr_center_y - other_centers_y)**2)
        
        # 【重要】五線幅の60%より近いものは「同じ音符への重複反応」とみなして削除
        # ただし、和音（縦の並び）を殺さないよう、x座標のズレが非常に小さい場合のみ縦距離を厳しく見る
        x_dist = np.abs(curr_center_x - other_centers_x)
        y_dist = np.abs(curr_center_y - other_centers_y)
        
        # 同じ音符判定：中心が近すぎる、または横位置がほぼ同じで縦も近すぎる
        duplicate = (dist < staff_space * 0.6) | ((x_dist < staff_space * 0.3) & (y_dist < staff_space * 0.7))
        
        idxs = np.delete(remaining_idxs, np.where(duplicate)[0])
        # deleteによってインデックスがズレるのを防ぐため、元のidxsリストを更新
        idxs = idxs[~np.isin(idxs, remaining_idxs[duplicate])] 

    return boxes[pick].astype("int")

def detect_note_heads_v7(gray_img, staff_space, user_threshold):
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 【新兵器】クロージング処理：白抜き音符の中を塗りつぶす
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(staff_space*0.6), int(staff_space*0.6)))
    filled_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
    
    # ノイズ除去
    k_size = max(2, int(staff_space * 0.3))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_thresh = cv2.morphologyEx(filled_thresh, cv2.MORPH_OPEN, kernel)
    
    nw, nh = int(staff_space * 1.2), int(staff_space * 0.9)
    template = np.zeros((nh + 4, nw + 4), dtype=np.uint8)
    cv2.ellipse(template, (nw//2+2, nh//2+2), (nw//2, nh//2), -20, 0, 360, 255, -1)

    res = cv2.matchTemplate(clean_thresh, template, cv2.TM_CCOEFF_NORMED)
    
    loc = np.where(res >= user_threshold)
    scores = res[loc]
    rects = [[int(pt[0]), int(pt[1]), int(pt[0]+nw), int(pt[1]+nh)] for pt in zip(*loc[::-1])]
    
    if not rects: return []
    return nms_v7_centroid(np.array(rects), scores, staff_space)

def get_pitch_name(note_y, staff, clef, flats_count):
    line1, line5 = staff[0], staff[4]
    step_size = abs(line1 - line5) / 8.0
    steps = int(round((line1 - note_y) / step_size))

    if clef == "treble":
        mapping = { -4:"ラ",-3:"シ",-2:"ド",-1:"レ", 0:"ミ",1:"ファ",2:"ソ",3:"ラ",4:"シ",5:"ド",6:"レ",7:"ミ",8:"ファ",9:"ソ",10:"ラ",11:"シ",12:"ド",13:"レ",14:"ミ",15:"ファ",16:"ソ" }
    else: # bass
        mapping = { -6:"ド",-5:"レ",-4:"ミ",-3:"ファ",-2:"ソ",-1:"ラ", 0:"シ",1:"ド",2:"レ",3:"ミ",4:"ファ",5:"ソ",6:"ラ",7:"シ",8:"ド",9:"レ",10:"ミ",11:"ファ",12:"ソ",13:"ラ",14:"シ" }

    name = mapping.get(steps, "")
    flat_order = ["シ", "ミ", "ラ", "レ", "ソ", "ド", "ファ"]
    return name, name in flat_order[:flats_count]

def process_page_v7(pil_img, threshold, flats_count):
    staves, space = detect_staff_groups_v7(pil_img)
    if not staves: return pil_img
    img_gray = np.array(pil_img.convert('L'))
    notes = detect_note_heads_v7(img_gray, space, threshold)
    result = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    try: font = ImageFont.truetype("NotoSansJP-Regular.ttf", max(15, int(space)))
    except: font = ImageFont.load_default()

    for box in notes:
        b = box.tolist()
        yc = (b[1] + b[3]) // 2
        s_idx = np.argmin([abs(np.mean(s) - yc) for s in staves])
        p_name, is_flat = get_pitch_name(yc, staves[s_idx], "treble" if s_idx % 2 == 0 else "bass", flats_count)
        if p_name:
            draw.rectangle(b, outline=(0, 255, 0), width=2)
            draw.text((b[0], b[1] - int(space * 1.6)), p_name, font=font, fill=((0,0,255) if is_flat else (255,0,0)))
    return result

# Streamlit UI
st.set_page_config(page_title="ドレミ付与 V7", layout="centered")
st.title("🎼 ドレミ付与ツール V7")
st.write("白抜き音符を自動で穴埋め検出し、重複を重心距離でカットする最新版です。")

st.sidebar.header("⚙️ 設定")
flats = st.sidebar.selectbox("調号（♭）の数", range(8), index=4) 
sens = st.sidebar.slider("検出感度", 0.2, 0.9, 0.45, 0.01)

up = st.file_uploader("PDFをアップロード", type="pdf")
if up:
    imgs = convert_from_bytes(up.read())
    with st.spinner('解析中...'):
        for im in [process_page_v7(i, sens, flats) for i in imgs]:
            st.image(im, use_column_width=True)

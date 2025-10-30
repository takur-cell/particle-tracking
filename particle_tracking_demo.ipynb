import cv2, numpy as np, os, csv, heapq

def calculate_average_background(background_dir):
    files = [f for f in os.listdir(background_dir) if f.lower().endswith(".bmp")]
    if not files: raise ValueError("背景画像が見つかりません。")
    acc = None
    for fn in files:
        img = cv2.imread(os.path.join(background_dir, fn), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        if acc is None:
            acc = np.zeros_like(img, dtype=np.float32)
        cv2.accumulate(img, acc)
    acc /= len(files)
    return np.clip(acc, 0, 255).astype(np.uint8)

def process_images(original_image_path, avg_background_u8, output_dir, writer, brightness=50):
    orig = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(f"オリジナル画像が見つかりません: {original_image_path}")

    bg = avg_background_u8
    if orig.shape != bg.shape:
        bg = cv2.resize(bg, (orig.shape[1], orig.shape[0]))

    # 元コードの意図「背景へ +brightness」を保ったまま、高速化
    bg = cv2.convertScaleAbs(bg, alpha=1.0, beta=brightness)     # 背景だけ +brightness
    inv_bg = cv2.bitwise_not(bg)                                  # 255 - (bg + brightness)
    subtracted = cv2.addWeighted(orig, 0.5, inv_bg, 0.5, 0)       # (orig - (bg+brightness) + 255)/2

    _, binary = cv2.threshold(subtracted, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積上位2個だけ部分選抜
    contours = heapq.nlargest(2, contours, key=cv2.contourArea)

    out = cv2.cvtColor(subtracted, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(contours):
        cv2.drawContours(out, [c], -1, (0,255,0), 2)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(out, f"Particle {i+1}", (cX, cY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            writer.writerow({"Particle": f"Particle {i+1}",
                             "X": cX, "Y": cY,
                             "Image": os.path.basename(original_image_path)})

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"processed_{os.path.basename(original_image_path)}"), out)

def process_all_images_in_directory(original_dir, background_dir, output_dir, csv_output_path):
    avg_background = calculate_average_background(background_dir)  # ★一度だけ計算
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Particle", "X", "Y", "Image"])
        writer.writeheader()
        for entry in os.scandir(original_dir):
            if entry.is_file() and entry.name.lower().endswith(".bmp"):
                try:
                    process_images(entry.path, avg_background, output_dir, writer)
                    print(f"処理完了: {entry.name}")
                except Exception as e:
                    print(f"エラー発生: {entry.name}, {e}")

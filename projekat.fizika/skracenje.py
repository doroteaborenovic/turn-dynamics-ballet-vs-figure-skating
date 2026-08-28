import pandas as pd
import numpy as np
import cv2
import os

# Putanja do fajla koji već imaš
csv_path = r"C:\Users\PC\gitara\projekat.fizika\obradjene_koordinate\koordinate_kamilavalieva_naucan.csv"
output_csv_dir = "konacne_koordinate"
output_vid_dir = "konacni_videi"
os.makedirs(output_csv_dir, exist_ok=True)
os.makedirs(output_vid_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# Sečenje tačno od 2. do 6. sekunde (odnosno prvih 121 frejm, od 0 do 120)
# Pošto je FPS obično 30, 2s = 60. frejm, do 6s = 180. frejm
start_frame = 60 
end_frame = 180

df_cropped = df[(df['Frame'] >= start_frame) & (df['Frame'] <= end_frame)].copy()
df_cropped['Original_Frame'] = df_cropped['Frame']
df_cropped['Frame'] = df_cropped['Frame'] - start_frame
df_cropped['Time'] = df_cropped['Frame'] / 30.0
df_cropped['FPS'] = 30.0

# Snimanje u konacne_koordinate
final_csv_path = os.path.join(output_csv_dir, "kamilavalieva_konacne.csv")
df_cropped.to_csv(final_csv_path, index=False, float_format='%.6f')

print(MI := f"Fajl uspešno isečen i sačuvan na: {final_csv_path}")
import pandas as pd
import glob
import os

folder_path = r"C:\Users\PC\Videos\Screen Recordings\processed_coordinates"
all_files = glob.glob(os.path.join(folder_path, "koordinate*.csv"))

print(f"Pronađeno fajlova: {len(all_files)}\n")

for file in all_files:
    df = pd.read_csv(file)
    print(f"Fajl: {os.path.basename(file)}")
    print(f" - Ukupan broj redova: {len(df)}")
    print(f" - Broj jedinstvenih frejmova: {df['Frame'].nunique()}")
    print(f" - Broj jedinstvenih Landmark ID-ijeva: {df['Landmark_ID'].nunique()} (očekivano je 33)")
    
    # Provera da li ima previše NaN (praznih) vrednosti
    nan_count = df[['X_world', 'Y_world', 'Z_world']].isna().sum().sum()
    print(f" - Broj praznih (NaN) koordinata: {nan_count}")
    
    # Provera opsega (MediaPipe World koordinate su u metrima, centrirane oko kuka)
    x_min, x_max = df['X_world'].min(), df['X_world'].max()
    print(f" - Opseg X_world: [{x_min:.2f}, {x_max:.2f}] metara")
    print("-" * 50) 

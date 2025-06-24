import pandas as pd
from pathlib import Path

def convert_excel_date(x):
    try:
        if pd.isna(x) or str(x).lower() == 'na':
            return pd.NaT
        return pd.to_datetime(float(x), origin='1899-12-30', unit='D')
    except ValueError:
        return pd.NaT

# Load the original clinical data
df = pd.read_csv("/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned/Clinical_Data/231009_clinical_data.csv")

# Parse all date columns to datetime
visit_date_cols = {
    "t1": "dateV1",
    "t2": "dateV2",
    "t3": "dateV3",
    "t4": "dateV4"
}
for col in visit_date_cols.values():
    # df[col] = pd.to_datetime(df[col], origin='1899-12-30', unit='D') # coerce bad formats to NaT
    df[col] = df[col].apply(convert_excel_date)

# Keep relevant base columns
# base_columns = ['Eye_ID', 'Gender', 'AgeatConsent', 'SmokingPY']
# visit_suffixes = ['t1', 't2', 't3', 't4']
# lesion_size_cols = visit_suffixes
# lesion_count_cols = [v + 'n' for v in visit_suffixes]

# Convert Gender to binary: male → 0, female → 1
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})

# Mapping visit suffixes to visit folder names
visit_map = {
    't1': 'V01',
    't2': 'V02',
    't3': 'V03',
    't4': 'V04',
}

# Prepare the output structure
records = []

# Base LakeFS data path
base_path = Path("/home/simone.sarrocco/OMEGA_study/data/OMEGA_cleaned")

# Iterate over each row (each study eye)
for _, row in df.iterrows():
    eye_id = row['Eye_ID']
    patient_id = eye_id.split('_')[0]  # Assumes Eye_ID is like 'OMEGA01_OS'
    eye_side_code = eye_id.split('_')[1]    # 'L' or 'R'
    eye_side = 'R' if eye_side_code == 'OD' else 'L'
    age_at_consent = float(row['AgeatConsent'])
    baseline_date = row['dateV1']
        
    for suffix in ['t1', 't2', 't3', 't4']:
        visit_prefix = visit_map[suffix]
        # Look for matching visit folders (e.g., "V02", "V02 - drop out", etc.)
        visit_root = base_path / patient_id / eye_side
        matching_visits = sorted(visit_root.glob(f"{visit_prefix}*"), key=lambda p: ('-' in p.name, p.name))

        visit_date = row[visit_date_cols[suffix]]

        if pd.isna(visit_date):
            continue

        # Compute age at this visit
        years_since_baseline = (visit_date - baseline_date).days / 365.25
        age_at_visit = age_at_consent + years_since_baseline
        age_at_visit_normalized = age_at_visit / 100.0  # Normalize age to [0, 1] range

        # Locate image path
        visit_folder = matching_visits[0]
        oct_folder = visit_folder/"Spectralis_oct"
        # Expected image file (assumes exactly one .nii or .nii.gz file per folder)
        nii_files = list(oct_folder.glob("*.nii.gz*"))
            
        if not nii_files:
            continue  # Skip if no image found

        image_path = nii_files[0]
        image_uid = image_path.stem
        latent_path = Path(str(image_path).replace(".nii.gz", "_latent.npz"))

        record = {
            "Patient_ID": patient_id,
            "Eye_ID": eye_id,
            "Gender": row["Gender"],
            "AgeatVisit": age_at_visit_normalized,
            "SmokingPY": row["SmokingPY"],
            "image_uid": image_uid,
            "image_path": str(image_path),
            "latent_path": str(latent_path),
            "GA_lesion_size": row[suffix],
            "GA_lesion_count": row[suffix + 'n'],
            "visit_id": visit_folder.name,
        }

        records.append(record)
# Create output dataframe
df_out = pd.DataFrame(records)

# Patient-wise split: group by patient ID
patient_ids = sorted(df_out['Patient_ID'].unique())
num_patients = len(patient_ids)
n_train = int(0.8 * num_patients)  # 80% for training
n_val = int(0.1 * num_patients)  # 10% for validation

train_patients = patient_ids[:n_train]
val_patients = patient_ids[n_train:n_train + n_val]
test_patients = patient_ids[n_train + n_val:]

# Assign splits
def assign_split(patient_id):
    if patient_id in train_patients:
        return 'train'
    elif patient_id in val_patients:
        return 'valid'
    else:
        return 'test'

df_out["split"] = df_out["Patient_ID"].apply(assign_split)

# Save to new CSV
df_out.to_csv("/home/simone.sarrocco/OMEGA_study/BrLP/examples/dataset.csv", index=False)

print("✅ CSV file 'dataset.csv' created successfully.")
print(f' Patient splits: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test')

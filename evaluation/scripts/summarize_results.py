import os
import re
import pandas as pd
from pathlib import Path

# --- GLOBAL PATHS ---
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
SUMMARY_TXT = RESULTS_DIR / "model_comparison.txt"
SUMMARY_CSV = RESULTS_DIR / "summary_data.csv"

def parse_existing_txt_notes(txt_path):
    """Parses the existing text summary to map (Dataset, Model, Method) to notes."""
    notes_map = {}
    if not txt_path.exists():
        return notes_map
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_dataset = None
        for line in lines:
            # Track which dataset section we are currently under
            if line.strip().startswith("DATASET:"):
                current_dataset = line.replace("DATASET:", "").strip().lower()
                continue

            if '|' in line and 'Model' not in line and '---' not in line and not line.strip().startswith('='):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3 and current_dataset:
                    model = parts[0]
                    method = parts[1]
                    note = parts[2]
                    if note:
                        # Include dataset in the unique key mapping
                        notes_map[(current_dataset, model, method)] = note
    except Exception as e:
        print(f"Note: Could not recover notes from txt file: {e}")
    return notes_map

def parse_report(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        model_match = re.search(r"Model\s+:\s+(.+)", content)
        if not model_match: 
            print(f"Skipping {file_path.name}: No 'Model :' pattern found.")
            return None
        model_name = model_match.group(1).strip()

        # --- FIX: FLEXIBLE & CLEAN FILENAME PARSING ---
        # Look for text between the '~' divider and standard suffixes
        dataset_match = re.search(r"~(.+?)(?:_output|_filled_slots|_baseline|_test)", file_path.name)
        if not dataset_match:
            # Fallback if t5-small style file names don't use the '~' convention
            dataset_match = re.search(r"(.+?)(?:_output|_filled_slots|_baseline|_cross_entropy|_qaset_loss)", file_path.name)
            
        if not dataset_match:
            raw_name = file_path.stem
        else:
            raw_name = dataset_match.group(1).strip()

        # Identify Training Method
        training_method = "N/A"
        for method in ["GRPO", "QASetLoss", "qaset_loss", "cross_entropy", "CE"]:
            if method in raw_name or method in file_path.name:
                if method in ["QASetLoss", "qaset_loss"]:
                    training_method = "QASetLoss"
                elif method in ["cross_entropy", "CE"]:
                    training_method = "CE"
                else:
                    training_method = method
                break
        
        # --- EXTRACT METADATA / AUTOMATIC NOTES FROM FILENAME ---
        notes_list = []
        
        checkpoint_match = re.search(r"checkpoint_\d+", file_path.name)
        if checkpoint_match:
            notes_list.append(checkpoint_match.group(0))
        elif "_7k" in file_path.name:
            notes_list.append("7k")
            
        if "B200" in file_path.name:
            notes_list.append("B200")
            
        if "output_B" in file_path.name:
            notes_list.append("output_B")
        elif "greedy" in file_path.name or "gredy" in file_path.name:
            notes_list.append("greedy")
        elif "baseline" in file_path.name:
            notes_list.append("baseline")

        # Clean up the true dataset anchor name down to standard core names (e.g., passive_red)
        dataset_name = re.sub(
            r"_(GRPO|QASetLoss|qaset_loss|cross_entropy|CE|7k|B200|checkpoint_\d+|instruct|baseline|filled_slots)", 
            "", 
            raw_name, 
            flags=re.IGNORECASE
        ).strip("_").lower()

        # Handle tokenized tracking variations
        fname_lower = file_path.name.lower()
        if "tokenized" in fname_lower and "detokenized" not in fname_lower:
            model_name += " (tokenized)"
        if "detokenized" in fname_lower:
            notes_list.append("detokenized")

        metric_pattern = r"([a-zA-Z\s]+):\s+(\d+\.\d+)%\s+(\d+\.\d+)%\s+(\d+\.\d+)%"
        metrics = re.findall(metric_pattern, content)
        
        data = {
            "Model": model_name, 
            "Dataset": dataset_name, 
            "Method": training_method,
            "Notes": ", ".join(notes_list) if notes_list else "",
            "Source_File": file_path.name
        }
        
        for name, p, r, f1 in metrics:
            n = name.strip().lower()
            data[f"{n} p"] = f"{float(p):.2f}%"
            data[f"{n} r"] = f"{float(r):.2f}%"
            data[f"{n} f1"] = f"{float(f1):.2f}%"

        return data
    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    
    existing_csv_notes = {}
    if SUMMARY_CSV.exists():
        try:
            old_df = pd.read_csv(SUMMARY_CSV)
            if "Source_File" in old_df.columns and "Notes" in old_df.columns:
                old_df["Notes"] = old_df["Notes"].fillna("").astype(str).str.strip()
                existing_csv_notes = dict(zip(old_df["Source_File"], old_df["Notes"]))
        except Exception:
            pass

    historical_txt_notes = parse_existing_txt_notes(SUMMARY_TXT)

    raw_files = list(RESULTS_DIR.glob("*.txt"))
    entries = [parse_report(f) for f in raw_files if f.name != "model_comparison.txt"]
    entries = [e for e in entries if e is not None]
    
    if not entries:
        print("No valid results found to summarize.")
        return

    for entry in entries:
        src_file = entry["Source_File"]
        model = entry["Model"]
        method = entry["Method"]
        
        if src_file in existing_csv_notes and existing_csv_notes[src_file]:
            entry["Notes"] = existing_csv_notes[src_file]
        elif (entry["Dataset"], model, method) in historical_txt_notes:
            entry["Notes"] = historical_txt_notes[(entry["Dataset"], model, method)]

    df = pd.DataFrame(entries)
    df = df.rename(columns=lambda x: x.strip().lower())
    
    col_order_lower = [
        "model", "method", "notes", 
        "unlabelled argument f1", "unlabelled argument p", "unlabelled argument r",
        "labelled argument f1", "labelled argument p", "labelled argument r",
        "unlabelled role f1", "unlabelled role p", "unlabelled role r"
    ]
    
    display_headers = [
        "Model", "Method", "Notes", 
        "Unlabelled Argument F1", "Unlabelled Argument P", "Unlabelled Argument R",
        "Labelled Argument F1", "Labelled Argument P", "Labelled Argument R",
        "Unlabelled Role F1", "Unlabelled Role P", "Unlabelled Role R"
    ]

    for col in col_order_lower:
        if col not in df.columns:
            df[col] = ""

    df.to_csv(SUMMARY_CSV, index=False)

    with open(SUMMARY_TXT, "w", encoding='utf-8') as f:
        f.write("QA-SRL MODEL EVALUATION SUMMARY\n")
        f.write("=" * 190 + "\n\n")

        # Safely convert F1 string to numeric float for accurate table ordering
        df['f1_numeric'] = df['unlabelled argument f1'].str.replace('%','', regex=False).replace('', '0').astype(float)
        
        # Sort values: Dataset grouped, then Model name, then Highest F1 score first
        df = df.sort_values(
            by=["dataset", "model", "f1_numeric"], 
            ascending=[True, True, False]
        )

        for dataset, group in df.groupby("dataset", sort=False):
            f.write(f"DATASET: {str(dataset).upper()}\n")
            f.write("-" * 30 + "\n")
            
            subset = group[col_order_lower].copy()
            subset["notes"] = subset["notes"].fillna("").astype(str)
            
            # Dynamic grid width calculations
            widths = {col: max(subset[col].apply(lambda x: len(str(x))).max(), len(display_headers[i])) + 2 
                      for i, col in enumerate(col_order_lower)}
            
            header = "| ".join(display_headers[i].ljust(widths[col]) for i, col in enumerate(col_order_lower))
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            prev_model = None
            for _, row in subset.iterrows():
                current_model = str(row["model"])
                if prev_model is not None and current_model != prev_model:
                    f.write("-" * len(header) + "\n")
                
                row_str = "| ".join(str(row[col]).ljust(widths[col]) for col in col_order_lower)
                f.write(row_str + "\n")
                prev_model = current_model
            
            f.write("\n" + "=" * 190 + "\n\n")

    print(f"Summary updated: {SUMMARY_TXT}")

if __name__ == "__main__":
    main()
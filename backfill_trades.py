import sys
import csv
from pathlib import Path
sys.path.append(str(Path("c:/Users/juanm/Documents/TRADING/MARK III")))
import MetaTrader5 as mt5
from datetime import datetime, timezone

trades_dir = Path("c:/Users/juanm/Documents/TRADING/MARK III/trades")

if not mt5.initialize():
    print("MT5 init failed")
    sys.exit(1)

for file in trades_dir.glob("trades_*.csv"):
    print(f"Processing {file.name}...")
    with open(file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []

    if not fieldnames:
        continue
        
    needs_update = False
    
    if "entry_time" not in fieldnames:
        idx = fieldnames.index("lot_size") + 1
        fieldnames.insert(idx, "entry_time")
        fieldnames.insert(idx + 1, "exit_time")
        needs_update = True
        
    for row in rows:
        if not row.get("entry_time"):
            ticket = row.get("ticket")
            if not ticket:
                continue
            
            deals = mt5.history_deals_get(position=int(ticket))
            if deals and len(deals) >= 2:
                entry_time = datetime.fromtimestamp(deals[0].time, tz=timezone.utc).isoformat()
                exit_time = datetime.fromtimestamp(deals[-1].time, tz=timezone.utc).isoformat()
                row["entry_time"] = entry_time
                row["exit_time"] = exit_time
                needs_update = True
            elif row.get("timestamp_utc"): 
                 row["exit_time"] = row["timestamp_utc"]
                 row["entry_time"] = row["timestamp_utc"]
                 needs_update = True
                 
    if needs_update:
        with open(file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            print(f"Updated {file.name}")
    else:
        print(f"No update needed for {file.name}")

mt5.shutdown()

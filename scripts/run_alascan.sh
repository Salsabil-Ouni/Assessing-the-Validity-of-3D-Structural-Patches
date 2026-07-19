#!/bin/bash
# Run FoldX AlaScan under WSL in a space-free working dir, then copy results
# back into the project. Called as:  wsl bash scripts/run_alascan.sh
set -u

BIN="/mnt/d/data D/Downloads D/foldx_Linux/foldx_20261231"
ROTA="/mnt/d/data D/Downloads D/foldx_Linux/rotabase.txt"
PDB="/mnt/d/Desktop/test 2/structure/tp53_repaired.pdb"
RUN="/tmp/foldx_run"
OUTDEST="/mnt/d/Desktop/test 2/results/foldx_raw"

rm -rf "$RUN"
mkdir -p "$RUN" "$OUTDEST"
cp "$ROTA" "$RUN/rotabase.txt"
cp "$PDB"  "$RUN/tp53_repaired.pdb"
chmod +x "$BIN"

cd "$RUN" || { echo "cd failed"; exit 1; }
echo "Working dir: $(pwd)"
echo "Launching FoldX AlaScan ..."
"$BIN" --command=AlaScan --pdb=tp53_repaired.pdb > foldx_log.txt 2>&1
echo "FoldX exit code: $?"

echo "=== files produced ==="
ls -la "$RUN"

# copy all FoldX outputs back into the project results/foldx_raw
cp -f "$RUN"/*.fxout "$OUTDEST"/ 2>/dev/null
cp -f "$RUN"/foldx_log.txt "$OUTDEST"/ 2>/dev/null
echo "=== copied into $OUTDEST ==="
ls -la "$OUTDEST"

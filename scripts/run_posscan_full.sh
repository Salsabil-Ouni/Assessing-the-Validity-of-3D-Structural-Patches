#!/bin/bash
set -u
BIN="/mnt/d/data D/Downloads D/foldx_Linux/foldx_20261231"
ROTA="/mnt/d/data D/Downloads D/foldx_Linux/rotabase.txt"
PDB="/mnt/d/Desktop/test 2/structure/tp53_chainA_zinc.pdb"
POSFILE="/mnt/d/Desktop/test 2/scripts/ps_positions.txt"
RUN="/tmp/foldx_ps_full"
OUTDEST="/mnt/d/Desktop/test 2/results/foldx_raw"

POS=$(cat "$POSFILE" | tr -d '\r\n')
rm -rf "$RUN"; mkdir -p "$RUN" "$OUTDEST"
cp "$ROTA" "$RUN/rotabase.txt"
cp "$PDB"  "$RUN/tp53_chainA_zinc.pdb"
cd "$RUN" || exit 1

echo "PositionScan over $(echo "$POS" | tr ',' '\n' | wc -l) positions ..."
START=$(date +%s)
"$BIN" --command=PositionScan --pdb=tp53_chainA_zinc.pdb --positions="$POS" > log.txt 2>&1
RC=$?; END=$(date +%s)
echo "FoldX exit=$RC  elapsed=$(( (END-START)/60 )) min $(( (END-START)%60 )) s"

cp -f PS_*_scanning_output.txt "$OUTDEST/positionscan_scanning_output.txt" 2>/dev/null
echo "lines in output: $(wc -l < "$OUTDEST/positionscan_scanning_output.txt" 2>/dev/null)"
echo "DONE"

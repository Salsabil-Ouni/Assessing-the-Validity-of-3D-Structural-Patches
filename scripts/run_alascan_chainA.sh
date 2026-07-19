#!/bin/bash
set -u
BIN="/mnt/d/data D/Downloads D/foldx_Linux/foldx_20261231"
ROTA="/mnt/d/data D/Downloads D/foldx_Linux/rotabase.txt"
PDB="/mnt/d/Desktop/test 2/structure/tp53_chainA_zinc.pdb"
RUN="/tmp/foldx_as_cA"
OUTDEST="/mnt/d/Desktop/test 2/results/foldx_raw"
rm -rf "$RUN"; mkdir -p "$RUN" "$OUTDEST"
cp "$ROTA" "$RUN/rotabase.txt"
cp "$PDB"  "$RUN/tp53_chainA_zinc.pdb"
cd "$RUN" || exit 1
"$BIN" --command=AlaScan --pdb=tp53_chainA_zinc.pdb > log.txt 2>&1
echo "AlaScan exit=$?"
cp -f tp53_chainA_zinc_AS.fxout "$OUTDEST/chainA_AS.fxout" 2>/dev/null
echo "output lines: $(wc -l < tp53_chainA_zinc_AS.fxout 2>/dev/null)"
head -3 tp53_chainA_zinc_AS.fxout

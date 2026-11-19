#!/usr/bin/env bash
# Download UD Swedish-Talbanken (CC BY-SA 4.0): noun lexicon + real Swedish regression sentences.
# REF is pinned to the commit used for the reported results. Run from the project root.
set -euo pipefail
REF="${REF:-0185ae1fb4728ae501bf2a329b958e6c6acb9c8d}"
mkdir -p data/ud
for split in train dev test; do
  curl -fsSL "https://raw.githubusercontent.com/UniversalDependencies/UD_Swedish-Talbanken/${REF}/sv_talbanken-ud-${split}.conllu" \
    -o "data/ud/sv_talbanken-ud-${split}.conllu"
done
# 100 real sentences (6-20 words, no digits) from the UD dev file for the fluency (NLL) regression check.
grep '^# text = ' data/ud/sv_talbanken-ud-dev.conllu | sed 's/^# text = //' \
  | awk 'NF>=6 && NF<=20 && $0 !~ /[0-9]/' | head -100 > data/ppl_sentences.txt
echo "UD_Swedish-Talbanken @ ${REF}"
echo "nouns: data/ud/*.conllu ($(wc -l < data/ppl_sentences.txt) regression sentences in data/ppl_sentences.txt)"

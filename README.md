# All defaults — txt and jsonl saved next to the PDF
python parse_hansard.py data/parliament.pdf

# Custom txt location
python parse_hansard.py data/parliament.pdf -t output/extracted.txt

# Custom jsonl location
python parse_hansard.py data/parliament.pdf -o output/chunks.jsonl

# Both custom
python parse_hansard.py data/parliament.pdf -t output/extracted.txt -o output/chunks.jsonl
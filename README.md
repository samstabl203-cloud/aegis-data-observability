# Data Quality Guardian

A beginner-friendly Python project that audits retail transaction data for:
- Missing values
- Duplicate rows and duplicate order IDs
- Invalid date formats
- Inconsistent product categories
- Suspicious numeric values (for example, non-positive quantity/price and statistical outliers)

It also evaluates detection against **known injected errors** using a clean sample dataset and a separately corrupted dataset.

## Project files

```text
data_quality_guardian/
├── data/
│   ├── retail_clean.csv
│   └── retail_corrupted.csv
├── reports/                  # created when you run the tool
├── src/
│   └── guardian.py
├── tests/
│   └── test_guardian.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer
- `pandas`
- `numpy`

## Quick start

Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt**
```bat
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the audit on the corrupted sample:

```bash
python src/guardian.py --input data/retail_corrupted.csv --out reports
```

Run the evaluation against the injected-error manifest:

```bash
python src/guardian.py --input data/retail_corrupted.csv --out reports --evaluate data/injected_errors.json
```

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

## Output files

The `reports/` directory will contain:
- `quality_summary.json` — issue counts and overall record count
- `issues.csv` — row-level findings, rule name, column, and explanation
- `cleaned_review_copy.csv` — a review copy with safe, limited normalization applied

**Important:** The tool does not silently delete rows or replace missing values. Its “cleaned review copy” only trims whitespace and normalizes known category spelling/case variants; it preserves the original input and flags other issues for human review.

## Detection rules and limitations

- Missing values: flags blank cells and common missing markers such as `NA`, `N/A`, `NULL`, and `None`.
- Duplicates: flags exact duplicate rows and repeated non-empty `order_id` values. Repeated order IDs may be legitimate in some data models, so review the rule before using it on other datasets.
- Date formats: expects `YYYY-MM-DD` in `order_date`.
- Categories: flags values outside the approved category list and recognizes a few common variants such as ` home decor ` and `HOME DECOR`.
- Numeric values: flags non-positive `quantity` and `unit_price`; also flags IQR outliers. Outliers are **suspicious, not automatically wrong**.
- Evaluation: compares detected row/column/rule findings to a manifest of injected errors. Since some rules intentionally detect broader patterns (such as IQR outliers), precision and recall are measured on the injected-error set and should be interpreted as a test of this sample, not a universal performance guarantee.

## Suggested next improvements

1. Add configurable column rules for other dataset schemas.
2. Add a dashboard using Streamlit.
3. Add data-quality trends across repeated daily imports.
4. Test on a public dataset and document any false positives.
5. Add a configuration file so category lists and numeric bounds can be changed without editing code.

## Data

`retail_clean.csv` is synthetic sample data created for this project. It does not contain real customer information. `retail_corrupted.csv` is generated from the clean sample with known errors for testing.

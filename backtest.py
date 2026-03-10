name: 极简回测引擎

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  run-backtest:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run backtest
        run: python backtest.py > summary.txt

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: backtest-report
          path: |
            summary.txt
            report.csv
            equity_curves.csv

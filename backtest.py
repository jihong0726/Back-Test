report, bench, equity_table = run_elite_tournament(df, fee_rate=0.0006, slippage=0.0002)

console_report = report.rename(columns={
    "选手流派": "Strategy",
    "净收益(%)": "Return%",
    "最大回撤(%)": "MaxDD%",
    "Sharpe": "Sharpe",
    "开仓次数": "Entries",
    "反手次数": "Flip",
    "持仓占比(%)": "Exposure%",
    "胜率(%)": "WinRate%",
    "盈亏比": "PnLRatio",
}).copy()

for col in ["Return%", "MaxDD%", "Exposure%", "WinRate%"]:
    console_report[col] = console_report[col].map(lambda x: f"{x:.2f}%")

console_report["Sharpe"] = console_report["Sharpe"].map(lambda x: f"{x:.2f}")
console_report["PnLRatio"] = console_report["PnLRatio"].map(lambda x: f"{x:.2f}")

print(f"\n[{VERSION}] Top Strategies Backtest")
print(f"Benchmark: {bench:.2f}%")
print("-" * 120)
print(console_report.to_string(index=False))

report.to_csv("report.csv", index=False, encoding="utf-8-sig")
equity_table.to_csv("equity_curves.csv", index=False, encoding="utf-8-sig")

with open("summary.md", "w", encoding="utf-8-sig") as f:
    f.write(f"# {VERSION} 回测结果\n\n")
    f.write(f"- Benchmark: {bench:.2f}%\n\n")
    f.write(report.to_markdown(index=False))

print("\n已输出 report.csv、equity_curves.csv、summary.md")

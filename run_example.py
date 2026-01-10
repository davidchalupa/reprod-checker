# run_example.py
from reprod_checker import ReprodChecker
from toy_train import train_fn

def main():
    checker = ReprodChecker(
        train_fn=train_fn,
        runs=4,
        base_seed=100,
        device="cpu",
        deterministic=True,
        save_models=False,
    )
    report = checker.run()
    report.pretty_print()
    report.save(".repro_report.json")
    print("Saved JSON report to .repro_report.json")

if __name__ == "__main__":
    main()

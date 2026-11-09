from schemesathi.data import write_quality_report


if __name__ == "__main__":
    report = write_quality_report()
    print(f"Dataset quality report written for {report['record_count']} records.")

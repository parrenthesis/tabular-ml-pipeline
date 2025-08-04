import pathlib
import shutil
import subprocess
import sys
from src.main import run_pipeline

# Ensure src/ is importable
project_root = pathlib.Path(__file__).resolve().parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_main_smoke():
    test_output_dir = "test_outputs"
    test_log_dir = "test_logs"
    try:
        run_pipeline(
            db_path="data/data.db",
            model="rf",
            output_dir=test_output_dir,
            log_dir=test_log_dir,
        )
        assert (
            pathlib.Path(test_output_dir) / "feature_importances_RandomForest.png"
        ).exists()
        assert (pathlib.Path(test_output_dir) / "pr_curve_RandomForest.png").exists()
    finally:
        # Clean up test outputs
        if pathlib.Path(test_output_dir).exists():
            shutil.rmtree(test_output_dir)
        if pathlib.Path(test_log_dir).exists():
            shutil.rmtree(test_log_dir)


def test_main_cli_entrypoint():
    # Run the CLI entrypoint with test args
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main",
            "--db-path",
            "data/data.db",
            "--model",
            "rf",
            "--output-dir",
            "test_outputs_cli",
            "--log-dir",
            "test_logs_cli",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "RandomForest" in result.stdout or "RandomForest" in result.stderr
    # Clean up
    out_dir = pathlib.Path("test_outputs_cli")
    log_dir = pathlib.Path("test_logs_cli")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if log_dir.exists():
        shutil.rmtree(log_dir)

"""
Tests for data fetch CLI modules.
"""

import pytest
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


class TestFetchCLI:
    """Test cases for fetch CLI functions."""
    
    def test_gee_fetch_help(self):
        """Test that GEE fetch script shows help when no arguments provided."""
        result = subprocess.run(
            [sys.executable, "src/data_fetch/fetch_gee.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "aoi" in result.stdout.lower()
        assert "start" in result.stdout.lower()
        assert "end" in result.stdout.lower()
    
    def test_appeears_fetch_help(self):
        """Test that AppEEARS fetch script shows help when no arguments provided."""
        result = subprocess.run(
            [sys.executable, "src/data_fetch/fetch_appeears.py", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "aoi" in result.stdout.lower()
        assert "start" in result.stdout.lower()
        assert "end" in result.stdout.lower()
    
    def test_gee_fetch_missing_args(self):
        """Test that GEE fetch script fails with appropriate error when required args missing."""
        result = subprocess.run(
            [sys.executable, "src/data_fetch/fetch_gee.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "required" in result.stderr.lower()
    
    def test_appeears_fetch_missing_args(self):
        """Test that AppEEARS fetch script fails with appropriate error when required args missing."""
        result = subprocess.run(
            [sys.executable, "src/data_fetch/fetch_appeears.py"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "required" in result.stderr.lower()


if __name__ == '__main__':
    pytest.main([__file__])

"""Exercise the real launcher branches while substituting only process launch."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == "nt" and shutil.which("powershell.exe"), "Windows launcher test")
class WindowsLauncherTests(unittest.TestCase):
    def run_launcher(self, *, exits=False):
        script = Path(__file__).resolve().parents[1] / "scripts" / "manage.ps1"
        # Keep the real path/dependency checks. Never launch an app or train here.
        harness = r'''
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle,
          $RedirectStandardOutput, $RedirectStandardError, [switch]$PassThru)
    if ((Split-Path -Leaf $FilePath) -ne 'pythonw.exe') { throw 'Expected windowed Python' }
    if ($WindowStyle -ne 'Normal') { throw 'Interactive Qt window must not be hidden' }
    if (-not $PassThru) { throw 'Startup exit detection missing' }
    if (-not $RedirectStandardError.EndsWith('.error.log')) { throw 'Startup error log missing' }
    if ($ArgumentList -ne ('"' + (Join-Path $WorkingDirectory 'app.py') + '"')) { throw 'Application path must be quoted' }
    [Console]::WriteLine('Verified visible windowed Python launch')
    $fakeProcess = [pscustomobject]@{ ExitCode = 17 }
    $fakeProcess | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value {
        param($timeout)
        if ($timeout -ne 3000) { throw 'Unexpected startup timeout' }
        return EXIT_RESULT
    }
    return $fakeProcess
}
& 'SCRIPT_PATH' -Action Start
exit $LASTEXITCODE
'''.replace("EXIT_RESULT", "$true" if exits else "$false").replace("SCRIPT_PATH", str(script).replace("'", "''"))
        with tempfile.TemporaryDirectory(prefix="launcher test ") as folder:
            path = Path(folder) / "launch test.ps1"
            path.write_text(harness, encoding="utf-8")
            return subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(path)],
                cwd=folder, capture_output=True, text=True, timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )

    def test_launch_is_visible_and_independent_of_current_directory(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Verified visible windowed Python launch", result.stdout)

    def test_early_crash_returns_failure_with_log_location(self):
        result = self.run_launcher(exits=True)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("application exited during startup", result.stdout)
        self.assertIn(".error.log", result.stdout)

    def test_batch_entry_points_choose_different_actions(self):
        root = Path(__file__).resolve().parents[1]
        start = (root / "Start.cmd").read_text()
        check = (root / "Check.cmd").read_text()
        self.assertIn('-Action Start', start)
        self.assertIn('-Action Check', check)
        self.assertIn('if not "%result%"=="0" pause', start)


if __name__ == "__main__":
    unittest.main()

import subprocess
import os

class ValidationCheckers:
    @staticmethod
    def verify_python_file(filepath):
        """Checks if a Python file is syntactically valid and runnable."""
        print(f"Verifying {filepath}...")
        try:
            # Run with python -m py_compile to check syntax without executing
            # Or just run it if it has a __main__ block (as requested in prompt)
            result = subprocess.run(
                ["python", filepath], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            if result.returncode == 0:
                print(f"✅ Verification Passed: {filepath}")
                return True, result.stdout
            else:
                print(f"❌ Verification Failed: {filepath}")
                print("Error Output:", result.stderr)
                return False, result.stderr
        except Exception as e:
            print(f"❌ Verification Error: {e}")
            return False, str(e)

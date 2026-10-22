"""Run the shared suite, optionally reusing an existing RoboCasa asset install.

Example: python cobot_integration/run_scenarios.py --robocasa-root /path/to/robocasa check ...
"""
import argparse
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--robocasa-root',type=Path)
    options,remaining=parser.parse_known_args()
    integration=Path(__file__).resolve().parent
    checkout=integration.parent
    sys.path.insert(0,str(checkout))
    if options.robocasa_root is not None:
        native=options.robocasa_root.resolve()
        if not (native/'robocasa/__init__.py').is_file():
            parser.error('--robocasa-root must contain robocasa/__init__.py')
        sys.path.insert(0,str(native))
    from cobot_scenarios.__main__ import main as suite_main
    sys.argv=[sys.argv[0],'--integration',str(integration),*remaining]
    suite_main()

if __name__=='__main__':
    main()

"""Verify the shareable package is complete without loading a simulator."""
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
from cobot_scenarios.core import validate

ROOT=Path(__file__).resolve().parents[2]

class PortableBundleTests(unittest.TestCase):
    def test_committed_curriculum_is_complete(self):
        data=ROOT/'cobot_scenarios/data'
        registry=json.loads((data/'registry.json').read_text())
        rows=json.loads((data/'scenario_index.json').read_text())
        self.assertEqual(len(rows),1000)
        self.assertTrue(validate(rows,registry))

    def test_robot_assets_are_bundled_and_relative(self):
        assets=ROOT/'cobot_integration/assets'
        for path in assets.glob('*.xml'):
            for element in ET.parse(path).getroot().iter():
                filename=element.get('file')
                if filename is None:continue
                relative=Path(filename)
                self.assertFalse(relative.is_absolute(),filename)
                target=(path.parent/relative).resolve()
                self.assertTrue(target.is_relative_to(assets.resolve()),filename)
                self.assertTrue(target.is_file(),str(target))

    def test_runtime_has_no_machine_specific_paths(self):
        for package in ('cobot_integration','cobot_scenarios'):
            for path in (ROOT/package).glob('*.py'):
                text=path.read_text()
                for prefix in ('/home/robocup/','/robodata/aditya/'):
                    self.assertNotIn(prefix,text,str(path))

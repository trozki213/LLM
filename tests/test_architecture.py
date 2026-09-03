"""The dependency rules from the design, enforced rather than asserted."""
import ast
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "fitkit"
DOMAIN = SRC / "domain"
STDLIB = set(sys.stdlib_module_names)


def _modules(root: pathlib.Path):
    return sorted(p for p in root.rglob("*.py"))


def _imports(path: pathlib.Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, stays inside the package
                yield "fitkit.domain", node.lineno
            elif node.module:
                yield node.module, node.lineno


class TestDomainIsSelfContained:
    def test_domain_imports_only_the_standard_library_and_itself(self):
        offenders = []
        for path in _modules(DOMAIN):
            for name, lineno in _imports(path):
                root = name.split(".")[0]
                if root in STDLIB or name.startswith("fitkit.domain"):
                    continue
                offenders.append(f"{path.relative_to(SRC)}:{lineno} imports {name}")
        assert offenders == []

    def test_the_scan_actually_found_modules(self):
        """A walker over an empty tree passes vacuously; guard against that."""
        assert len(_modules(DOMAIN)) >= 8

    def test_domain_declares_no_third_party_dependency_in_the_project_metadata(self):
        pyproject = (SRC.parents[1] / "pyproject.toml").read_text()
        runtime = pyproject.split("dependencies = ")[1].split("\n")[0]
        assert runtime.strip() == "[]"


class TestNoBareCentimetres:
    """Outside the contract wire format, a centimetre is a Measure, never a float."""

    EXEMPT_FILES = {"units.py", "policy.py"}

    def test_no_public_float_field_is_named_like_a_measurement(self):
        offenders = []
        for path in _modules(DOMAIN):
            if path.name in self.EXEMPT_FILES or "contracts" in path.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
                    continue
                name = node.target.id
                annotation = ast.unparse(node.annotation)
                if name.endswith("_cm") and "float" in annotation and "Measure" not in annotation:
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} {name}: {annotation}")
        assert offenders == []

    def test_the_exemptions_are_real_files_with_stated_reasons(self):
        for name in self.EXEMPT_FILES:
            path = DOMAIN / name
            assert path.exists(), name
            assert "bare-cm-exempt:" in path.read_text(), f"{name} must state why it is exempt"


class TestNoAccidentalGlobalState:
    def test_domain_defines_no_module_level_mutable_state(self):
        offenders = []
        for path in _modules(DOMAIN):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:
                targets = []
                if isinstance(node, ast.Assign):
                    targets = [t for t in node.targets if isinstance(t, ast.Name)]
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    targets = [node.target]
                for t in targets:
                    if isinstance(node.value, (ast.Dict, ast.List, ast.Set)):
                        offenders.append(f"{path.relative_to(SRC)}:{node.lineno} {t.id}")
        assert offenders == []

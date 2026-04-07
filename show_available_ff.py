from pathlib import Path
import openmm.app as app

data_dir = Path(app.__file__).resolve().parent / "data"
for path in sorted(data_dir.rglob("*.xml")):
    print(path.relative_to(data_dir))


print('===========')
print(Path(app.__file__).resolve().parent / "data")

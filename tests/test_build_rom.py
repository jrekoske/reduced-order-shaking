import os
import yaml
import shutil
from romshake.core.numerical_rom_builder import NumericalRomBuilder


def test_build_rom():
    with open('test_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    os.makedirs('test')
    nrb = NumericalRomBuilder(**config)
    nrb.train()
    shutil.rmtree('test')
    shutil.rmtree('cachedir')


if __name__ == '__main__':
    os.environ['CALLED_FROM_PYTEST'] = 'True'
    test_build_rom()

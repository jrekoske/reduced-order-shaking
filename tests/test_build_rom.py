import os
import shutil
import subprocess


def test_build_rom():
    ret = subprocess.call(['build_rom', 'test_config.yaml'])
    shutil.rmtree('test')
    shutil.rmtree('cachedir')
    assert (ret == 0)


if __name__ == '__main__':
    os.environ['CALLED_FROM_PYTEST'] = 'True'
    test_build_rom()

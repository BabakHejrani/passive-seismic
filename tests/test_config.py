import pytest
import datetime
import yaml
import os
from seismic import config

TESTS = os.path.dirname(__file__)
DATA = os.path.join(TESTS, 'mocks', 'data')


@pytest.fixture(params=['miniseeds', 'events', 'times'])
def custom_conf(conf, params_dict, request):
    conf['inputs'].append(params_dict[request.param])
    return conf


@pytest.fixture(params=['miniseeds', 'events', 'times'])
def custom_conf_2s(conf, params_dict, request):
    all_supported_formats = ['miniseeds', 'events', 'times']
    all_supported_formats.remove(request.param)
    conf['inputs'] += [params_dict[a] for a in all_supported_formats]
    return conf


def test_conflict_2s(random_filename, custom_conf_2s):
    yaml_file = random_filename(ext='.yaml')
    with open(yaml_file, 'w') as outfile:
        yaml.dump(custom_conf_2s, outfile, default_flow_style=False)
    with pytest.raises(config.ConfigException, message='Too many inputs '
                                                       'provided'):
        config.Config(yaml_file)


def test_conflict(random_filename, conf, params_dict):
    conf['inputs'] += [params_dict[a]
                       for a in ['miniseeds', 'events', 'times']]
    yaml_file = random_filename(ext='.yaml')
    with open(yaml_file, 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)
    with pytest.raises(config.ConfigException, message='Too many inputs '
                                                       'provided'):
        config.Config(yaml_file)


def test_no_conflict(random_filename, custom_conf):
    yaml_file = random_filename(ext='.yaml')
    with open(yaml_file, 'w') as outfile:
        yaml.dump(custom_conf, outfile, default_flow_style=False)
    try:
        config.Config(yaml_file)
    except config.ConfigException:
        pytest.fail(msg='Too many inputs provided')


@pytest.fixture(params=[3, 1, 12])
def month(request):
    return request.param


@pytest.fixture(params=[1, 5, 18, 28])
def day(request):
    return request.param


def test_dateformat(random_filename, conf, month, day):
    times = {
        'times': {
            'end_time': datetime.datetime(2017, month, day, 17, 18, 30),
            'start_time': datetime.datetime(2017, month, day, 16, 18, 30)},
        'name': 'my time range', 'type': 'time'
    }

    conf['inputs'].append(times)
    yaml_file = random_filename(ext='.yaml')
    with open(yaml_file, 'w') as outfile:
        yaml.dump(conf, outfile, default_flow_style=False)

    cf = config.Config(yaml_file)
    assert 'start_time' in cf.time_range
    assert 'end_time' in cf.time_range
    assert cf.time_range['start_time'].month == month
    assert cf.time_range['start_time'].day == day
    assert cf.time_range['end_time'].month == month
    assert cf.time_range['end_time'].day == day

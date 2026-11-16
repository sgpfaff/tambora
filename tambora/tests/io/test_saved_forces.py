"""The external forces a run file keeps, including each force's label and the potentials saved with the run."""

import io
import warnings

import galpy
import h5py
import numpy as np
import pytest
from galpy import potential as gp

from tambora.dynamics import ExternalPotential
from tambora.interop._galpy.potential import GalpyPotential
from tambora.io.run_files import _read_settings, _write_settings, read_run
from tambora.simulation import Sim
from tambora.tests.io.test_save import _same
from tambora.tests.io.test_streaming_sim import RUN, _Drag, _sim

POS = np.array([[8., 0., 0.5], [3., -2., 1.], [0.5, 4., -2.]])   # kpc
NOT_SAVED = "can't be saved with the run"


def _nfw():
    return gp.NFWPotential(a=2., normalize=0.35, ro=8., vo=220.)


def _ramp():
    """A potential holding a lambda, which pickle can't save."""
    return gp.TimeDependentAmplitudeWrapperPotential(A=lambda t: 1., pot=gp.NFWPotential(ro=8., vo=220.))


def test_a_run_file_keeps_each_force_in_order(tmp_path):
    sim = _sim(output_dir=tmp_path / 'run')
    sim.add_external_force(_Drag())
    sim.add_external_pot(_nfw())
    sim.add_external_pot(gp.MWPotential2014)
    sim.run(**RUN)
    run = read_run(tmp_path / 'run')
    assert [f['label'] for f in run['forces']] == run['settings']['run']['forces']
    nfw, mw, drag = run['forces']
    assert drag == {'label': '_Drag', 'backend': None, 'saved': False, 'error': None}
    for record, force in zip((nfw, mw), sim._conserv_ext_force.members):
        assert (record['backend'], record['saved'], record['error']) == ('galpy', True, None)
        assert record['record']['galpy_version'] == galpy.__version__
        loaded = GalpyPotential.load(record['record'])
        np.testing.assert_array_equal(loaded.acc(POS, 0.1), force.acc(POS, 0.1))


def test_a_saved_run_keeps_the_records_a_streamed_run_does(tmp_path):
    for sim in (_sim(output_dir=tmp_path / 'streamed'), _sim()):
        sim.add_external_pot(_nfw())
        sim.add_external_pot(gp.MWPotential2014)
        sim.add_external_force(_Drag())
        sim.run(**RUN)
    sim.save(tmp_path / 'saved')
    Sim.load(tmp_path / 'streamed').save(tmp_path / 'copy of loaded')
    expected = read_run(tmp_path / 'streamed')['forces']
    for copy in ('saved', 'copy of loaded'):
        assert _same(read_run(tmp_path / copy)['forces'], expected)


def test_a_potential_is_kept_as_it_was_when_the_run_started(tmp_path):
    nfw = _nfw()
    sim = _sim()
    sim.add_external_pot(nfw)
    sim.run(**RUN)
    during = sim._conserv_ext_force.members[0].acc(POS, 0.1)
    nfw._amp *= 2.
    sim.save(tmp_path / 'saved')
    record = read_run(tmp_path / 'saved')['forces'][0]['record']
    np.testing.assert_array_equal(GalpyPotential.load(record).acc(POS, 0.1), during)
    sim.run(**RUN)                          # a rerun keeps the potential as it is now
    np.testing.assert_allclose(GalpyPotential.load(sim._run_forces[0]['record']).acc(POS, 0.1), 2. * during)


@pytest.mark.parametrize("add", ['add_external_pot', 'add_external_force'])
def test_save_with_run_false_keeps_only_the_label_without_a_word(tmp_path, add):
    for sim, where in ((_sim(output_dir=tmp_path / 'streamed'), None), (_sim(), tmp_path / 'saved')):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            if add == 'add_external_pot':
                sim.add_external_pot(_ramp(), save_with_run=False)
            else:
                sim.add_external_force(ExternalPotential(_ramp(), save_with_run=False))
            sim.run(**RUN)
            if where is not None:
                sim.save(where)
    for where in ('streamed', 'saved'):
        assert read_run(tmp_path / where)['forces'] == [
            {'label': 'ExternalPotential(_TimeDependentAmplitudeWrapperPotential)', 'backend': 'galpy',
             'saved': False, 'error': None}]


@pytest.mark.parametrize("add", ['add_external_pot', 'add_external_force', 'a sum of forces'])
def test_a_streamed_sim_warns_when_a_potential_that_cant_be_saved_is_added(tmp_path, add):
    sim = _sim(output_dir=tmp_path / 'run')
    with pytest.warns(UserWarning, match=NOT_SAVED) as caught:
        if add == 'add_external_pot':
            sim.add_external_pot(_ramp())
        elif add == 'add_external_force':
            sim.add_external_force(ExternalPotential(_ramp()))
        else:
            sim.add_external_force(ExternalPotential(_nfw()) + ExternalPotential(_ramp()))
    assert len(caught) == 1 and caught[0].filename == __file__      # points at your line
    with warnings.catch_warnings():
        warnings.simplefilter('error')      # not again when it runs
        sim.run(**RUN)
    record = read_run(tmp_path / 'run')['forces'][-1]
    assert (record['saved'], record['error'].split(':')[0]) == (False, 'AttributeError')
    assert 'record' not in record


def test_an_in_memory_sim_warns_when_it_is_saved(tmp_path):
    sim = _sim()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sim.add_external_pot(_ramp())
        sim.run(**RUN)
    with pytest.warns(UserWarning, match=NOT_SAVED) as caught:
        sim.save(tmp_path / 'saved')
    assert len(caught) == 1 and caught[0].filename == __file__
    assert read_run(tmp_path / 'saved')['forces'][0]['error'].startswith('AttributeError')


def test_a_potential_that_stops_pickling_after_it_was_added_is_warned_about_when_the_run_starts(tmp_path):
    nfw = _nfw()
    sim = _sim(output_dir=tmp_path / 'run')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sim.add_external_pot(nfw)
    nfw.hook = lambda: None
    with pytest.warns(UserWarning, match=NOT_SAVED):
        sim.run(**RUN)
    assert read_run(tmp_path / 'run')['forces'][0]['saved'] is False


def test_a_loaded_sim_doesnt_warn_and_its_copy_says_why_the_potential_wasnt_saved(tmp_path):
    sim = _sim()
    sim.add_external_pot(_ramp())
    sim.run(**RUN)
    with pytest.warns(UserWarning, match=NOT_SAVED):
        sim.save(tmp_path / 'saved')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')     # the forces that couldn't be loaded
        loaded = Sim.load(tmp_path / 'saved')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        loaded.add_external_pot(_ramp())    # it can't run, so nothing would be written
        loaded.save(tmp_path / 'copy')
    assert _same(read_run(tmp_path / 'copy')['forces'], read_run(tmp_path / 'saved')['forces'])


@pytest.mark.parametrize("data", [b'pickle\x00with\x00nul bytes', b'', bytes(range(256)) * 100])
def test_bytes_in_a_settings_dict_come_back_as_written(data):
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as f:
        _write_settings(f.create_group('s'), {'data': data, 'nested': {'data': data}})
    with h5py.File(buffer, 'r') as f:
        assert _read_settings(f['s']) == {'data': data, 'nested': {'data': data}}

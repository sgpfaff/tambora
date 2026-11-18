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
    sim.add_force(_Drag())
    sim.add_force(_nfw())
    sim.add_force(gp.MWPotential2014)
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
        sim.add_force(_nfw())
        sim.add_force(gp.MWPotential2014)
        sim.add_force(_Drag())
        sim.run(**RUN)
    sim.save(tmp_path / 'saved')
    Sim.load(tmp_path / 'streamed').save(tmp_path / 'copy of loaded')
    expected = read_run(tmp_path / 'streamed')['forces']
    for copy in ('saved', 'copy of loaded'):
        assert _same(read_run(tmp_path / copy)['forces'], expected)


def test_a_potential_is_kept_as_it_was_when_the_run_started(tmp_path):
    nfw = _nfw()
    sim = _sim()
    sim.add_force(nfw)
    sim.run(**RUN)
    during = sim._conserv_ext_force.members[0].acc(POS, 0.1)
    nfw._amp *= 2.
    sim.save(tmp_path / 'saved')
    record = read_run(tmp_path / 'saved')['forces'][0]['record']
    np.testing.assert_array_equal(GalpyPotential.load(record).acc(POS, 0.1), during)
    sim.run(**RUN)                          # a rerun keeps the potential as it is now
    np.testing.assert_allclose(GalpyPotential.load(sim._run_forces[0]['record']).acc(POS, 0.1), 2. * during)


@pytest.mark.parametrize("wrap", [lambda pot: pot, ExternalPotential], ids=['a potential', 'a force'])
def test_save_with_run_false_keeps_only_the_label_without_a_word(tmp_path, wrap):
    for sim, where in ((_sim(output_dir=tmp_path / 'streamed'), None), (_sim(), tmp_path / 'saved')):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            sim.add_force(wrap(_ramp()), save_with_run=False)
            sim.run(**RUN)
            if where is not None:
                sim.save(where)
    for where in ('streamed', 'saved'):
        assert read_run(tmp_path / where)['forces'] == [
            {'label': 'ExternalPotential(_TimeDependentAmplitudeWrapperPotential)', 'backend': 'galpy',
             'saved': False, 'error': None}]


def test_save_with_run_false_holds_for_each_force_in_a_sum_and_only_for_them(tmp_path):
    sim = _sim(output_dir=tmp_path / 'run')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sim.add_force(ExternalPotential(_ramp()) + _Drag() + ExternalPotential(_nfw()), save_with_run=False)
    sim.add_force(gp.MiyamotoNagaiPotential(ro=8., vo=220.))
    sim.run(**RUN)
    assert [(f['label'], f['saved']) for f in read_run(tmp_path / 'run')['forces']] == [
        ('ExternalPotential(_TimeDependentAmplitudeWrapperPotential)', False), ('ExternalPotential(NFWPotential)', False),
        ('ExternalPotential(MiyamotoNagaiPotential)', True), ('_Drag', False)]


@pytest.mark.parametrize("add", ['a potential', 'a force', 'a sum of forces'])
def test_a_streamed_sim_warns_when_a_potential_that_cant_be_saved_is_added(tmp_path, add):
    sim = _sim(output_dir=tmp_path / 'run')
    with pytest.warns(UserWarning, match=NOT_SAVED) as caught:
        if add == 'a potential':
            sim.add_force(_ramp())
        elif add == 'a force':
            sim.add_force(ExternalPotential(_ramp()))
        else:
            sim.add_force(ExternalPotential(_nfw()) + ExternalPotential(_ramp()))
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
        sim.add_force(_ramp())
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
        sim.add_force(nfw)
    nfw.hook = lambda: None
    with pytest.warns(UserWarning, match=NOT_SAVED):
        sim.run(**RUN)
    assert read_run(tmp_path / 'run')['forces'][0]['saved'] is False


def test_a_loaded_sim_doesnt_warn_and_its_copy_says_why_the_potential_wasnt_saved(tmp_path):
    sim = _sim()
    sim.add_force(_ramp())
    sim.run(**RUN)
    with pytest.warns(UserWarning, match=NOT_SAVED):
        sim.save(tmp_path / 'saved')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        loaded = Sim.load(tmp_path / 'saved', forces=[_ramp()])     # it can't run, so nothing would be written
        loaded.save(tmp_path / 'copy')
    assert _same(read_run(tmp_path / 'copy')['forces'], read_run(tmp_path / 'saved')['forces'])


@pytest.mark.parametrize("data", [b'pickle\x00with\x00nul bytes', b'', bytes(range(256)) * 100])
def test_bytes_in_a_settings_dict_come_back_as_written(data):
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as f:
        _write_settings(f.create_group('s'), {'data': data, 'nested': {'data': data}})
    with h5py.File(buffer, 'r') as f:
        assert _read_settings(f['s']) == {'data': data, 'nested': {'data': data}}


# --- Sim.load adds the saved potentials again ------------------------------------------- #

def _run_with_potentials(directory):
    sim = _sim(output_dir=directory)
    sim.add_force(_nfw())
    sim.add_force(gp.MiyamotoNagaiPotential(ro=8., vo=220.))
    sim.add_force(_Drag())
    sim.run(**RUN)
    return sim


def test_a_loaded_sim_has_the_runs_potentials_in_order_and_computes_as_the_run_did(tmp_path):
    run = _run_with_potentials(tmp_path / 'run')
    with pytest.warns(UserWarning, match=r"^_Drag couldn't be loaded"):
        sim = Sim.load(tmp_path / 'run')
    assert [repr(f) for f in sim._conserv_ext_force.members] == [repr(f) for f in run._conserv_ext_force.members]
    for t in (0., 0.1):
        np.testing.assert_array_equal(sim._conserv_ext_force.acc(POS, None, t), run._conserv_ext_force.acc(POS, None, t))
    assert sim._not_saved == []


def test_giving_load_a_potential_the_run_saved_is_refused_as_a_duplicate(tmp_path):
    _run_with_potentials(tmp_path / 'run')
    with pytest.raises(ValueError, match="already added"):
        Sim.load(tmp_path / 'run', forces=[_nfw()])


def _loading_with(monkeypatch, change):
    """Sim.load reading run.h5's forces after *change* changed the first one's record."""
    import tambora.simulation.simulation as simulation_module
    read = simulation_module.read_run

    def changed(directory):
        run = read(directory)
        change(run['forces'][0])
        return run
    monkeypatch.setattr(simulation_module, 'read_run', changed)


@pytest.mark.parametrize("change, reason", [
    (lambda f: f['record'].update(data=b'not a pickle'), r"\(UnpicklingError: "),
    (lambda f: f.update(backend='agama'), r"\(ValueError: This tambora has no 'agama' potential backend.\)"),
], ids=['damaged', 'unknown backend'])
def test_a_potential_that_cant_be_loaded_is_named_with_why_and_the_rest_loads(tmp_path, monkeypatch, change, reason):
    run = _run_with_potentials(tmp_path / 'run')
    _loading_with(monkeypatch, change)
    with pytest.warns(UserWarning, match=r"^ExternalPotential\(NFWPotential\) " + reason) as caught:
        sim = Sim.load(tmp_path / 'run')
    assert len(caught) == 1 and caught[0].message.args[0].count("couldn't be loaded") == 1
    assert ", _Drag couldn't be loaded" in str(caught[0].message)
    assert [repr(f) for f in sim._conserv_ext_force.members] == [repr(run._conserv_ext_force.members[1])]


def test_a_potential_from_another_galpy_warns_from_your_line_and_loads(tmp_path, monkeypatch):
    run = _run_with_potentials(tmp_path / 'run')
    _loading_with(monkeypatch, lambda f: f['record'].update(galpy_version='0.1'))
    with pytest.warns(UserWarning) as caught:
        sim = Sim.load(tmp_path / 'run')
    version = [w for w in caught if 'saved with galpy 0.1' in str(w.message)]
    assert len(version) == 1 and version[0].filename == __file__
    assert [repr(f) for f in sim._conserv_ext_force.members] == [repr(f) for f in run._conserv_ext_force.members]


def test_a_potential_saved_without_its_package_installed_is_named(tmp_path, monkeypatch):
    from tambora.interop import _registry
    _run_with_potentials(tmp_path / 'run')
    monkeypatch.setattr(_registry, '_POTENTIAL_BACKENDS', (_registry._Entry('galpy', 'no_such_package_xyz', '', ''),))
    with pytest.warns(UserWarning, match=r"\(ImportError: no_such_package_xyz isn't installed.\)") as caught:
        sim = Sim.load(tmp_path / 'run')
    assert len(caught) == 1 and sim._conserv_ext_force.members == []


def test_a_potential_that_wasnt_saved_is_named_without_a_reason(tmp_path):
    sim = _sim(output_dir=tmp_path / 'run')
    sim.add_force(_nfw(), save_with_run=False)
    sim.run(**RUN)
    with pytest.warns(UserWarning, match=r"^ExternalPotential\(NFWPotential\) couldn't be loaded"):
        Sim.load(tmp_path / 'run')

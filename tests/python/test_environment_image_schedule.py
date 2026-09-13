"""Full-image ownership and aggregate work must be explicit before execution."""
import importlib
import pytest


def module(): return importlib.import_module('tools.environment_image_schedule')


def test_environment_schedule_partitions_every_pixel_under_both_limits():
    orders = [128]*40+[32]*150+[64]*9
    result = module().partition(orders)
    assert [i for row in result for i in range(row['start'], row['end'])] == list(range(len(orders)))
    assert all(row['calls'] <= 4096 and row['end']-row['start'] <= 128 for row in result)
    assert sum(row['calls'] for row in result) == 2*sum(max(1, 4*k*k//1024) for k in orders)
    assert sum(row['directions'] for row in result) == 2*sum(4*k*k for k in orders)


@pytest.mark.parametrize('orders', [[], [256], [True], [8.], [8]*6537])
def test_environment_schedule_rejects_unbounded_orders(orders):
    with pytest.raises(ValueError): module().partition(orders)


def test_environment_schedule_checks_progress_hash_before_json(tmp_path):
    path = tmp_path/'progress.json'; path.write_text('not json')
    with pytest.raises(ValueError, match='hash'):
        module().progress(path, '0'*64)


def test_environment_schedule_rebases_independent_colour_affine_reference():
    b, target = module().rebase([.6, .7, .8], [.2, .3, .4], [.8, .5, .9], [.1, .2, .3])
    assert b == pytest.approx([.44, .55, .44])
    assert target == pytest.approx([.52, .65, .71])

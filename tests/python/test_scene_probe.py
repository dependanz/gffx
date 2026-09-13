"""Keep CPU qualification from accidentally selecting GPU or benchmark tests."""
def test_scene_worker_selects_exact_nodes_not_filename_substrings():
    from tools.scene_probe import test_selection
    cpu=test_selection('integration');gpu=test_selection('cuda');budget=test_selection('budget')
    assert len(cpu)==3 and len(gpu)==len(budget)==1
    assert all('::test_scene_integration_' in n for n in cpu)
    assert all('::test_scene_cuda_' in n for n in gpu)
    assert all('::test_scene_budget_' in n for n in budget)
    assert not(set(cpu)&set(gpu) or set(cpu)&set(budget))

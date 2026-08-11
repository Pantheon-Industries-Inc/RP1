from rlp.campaigns.rlp_20260811 import jobs, validate


def test_campaign_matrix_matches_handoff_capacity() -> None:
    matrix = validate()
    assert len(matrix) == 16
    assert sum(job.gpus for job in matrix) == 64
    assert len([job for job in matrix if job.group == "tworoom"]) == 12
    assert {job.name for job in matrix} == {job.name for job in jobs()}

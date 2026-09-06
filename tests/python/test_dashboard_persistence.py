"""Regression coverage for dashboard queue tails (contract sections 2.4 and 9)."""

from gffx import dashboard as gd


class LegacyQueue:
    """Exercise historical protocol-1 senders that omit the new snapshot policy fields."""
    def __init__(self, client):
        self.client = client

    def push(self, value):
        self.client._send({"t": "push", "path": "fit/stream", "kind": "scalar", "bound": 4, "value": value}, [])


def test_queue_tail_restarts_preserve_steps_and_next_push(tmp_path):
    """Replaying successive tail snapshots must not invent queue steps."""
    expected = {}
    for restart in range(4):
        server = gd.Server(root=tmp_path, allow_loopback_for_tests=True, port=0).start()
        client = None
        try:
            if restart:
                server.replay("tail")
                actual = server.state.series("fit/stream", run="tail")
                assert {step: row["value"] for step, row in actual.items()} == expected
            client = gd.connect(server.address, run="tail", wait_acks=True)
            queue = LegacyQueue(client)
            values = range(10) if restart == 0 else [9 + restart]
            for value in values:
                queue.push(float(value))
                expected[value] = float(value)
                while len(expected) > 4:
                    del expected[min(expected)]
            actual = server.state.series("fit/stream", run="tail")
            assert {step: row["value"] for step, row in actual.items()} == expected
        finally:
            if client is not None:
                client.close()
            server.stop()


def test_push_is_unlogged_until_clean_shutdown(tmp_path):
    server = gd.Server(root=tmp_path, allow_loopback_for_tests=True, port=0).start()
    client = None
    try:
        client = gd.connect(server.address, run="tail", wait_acks=True)
        log = server.run_dir("tail") / "log.bin"
        before = log.read_bytes()
        queue = LegacyQueue(client)
        for value in range(10):
            queue.push(float(value))
        assert log.read_bytes() == before
    finally:
        if client is not None:
            client.close()
        server.stop()
    assert len(log.read_bytes()) > len(before)
    fresh = gd.Server(root=tmp_path, allow_loopback_for_tests=True, port=0)
    fresh.replay("tail")
    assert list(fresh.state.series("fit/stream", run="tail")) == [6, 7, 8, 9]

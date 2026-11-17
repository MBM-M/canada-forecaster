from service.worker import Settings, Worker


def drain(conn, raw, **settings) -> None:
    """Run one in-process worker until nothing is left."""
    Worker(conn, Settings(**settings), raw=raw).run_forever(idle_sleep=0.05, exit_when_idle=True)

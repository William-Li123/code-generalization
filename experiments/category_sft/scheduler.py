"""Dynamic, exclusive GPU lanes; no changes to a job's scientific commands."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def run_dynamic(jobs, run_job, workers=4, on_event=None):
    if workers < 1:
        raise ValueError('At least one worker is required')
    pending = iter(jobs)
    notify = on_event or (lambda *args: None)
    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        active = {}

        def dispatch(gpu):
            job = next(pending, None)
            if job is None:
                return
            notify('start', gpu, job)
            active[pool.submit(run_job, job, gpu)] = (gpu, job)

        for gpu in range(workers):
            dispatch(gpu)
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            ordered = sorted(done, key=lambda future: active[future][0])
            failure = None
            # Check all completed futures before dispatching more work.
            for future in ordered:
                gpu, job = active[future]
                error = future.exception()
                if error is not None:
                    notify('failed', gpu, job)
                    failure = failure or error
            if failure is not None:
                # Already-running jobs finish safely; no new job is started.
                raise failure
            for future in ordered:
                gpu, job = active.pop(future)
                future.result()
                finished += 1
                notify('done', gpu, job)
            for gpu in sorted(set(range(workers)) - {g for g, _ in active.values()}):
                dispatch(gpu)
    return finished

import time
from invoke import task, call
from timeit import timeit

@task
def clean(c, extra=''):
    patterns = ['*.so']
    if extra:
        patterns.append(extra)
    for pattern in patterns:
        c.run("rm -rf {}".format(pattern))
    print("Done cleaning")


# Same as: @task(pre=[clean]) - can only work on 'pre=' parameter
@task(clean)
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")
    print("Done building")


@task
def clear(c):
    c.run("clear")

@task
def sleep(c, seconds=3):
    time.sleep(seconds)


@task(pre=[clear], post=[call(sleep, 5), clear], aliases=['pz'])
def python_zen(c):
    """
    Usage: inv --no-dedupe python-zen
           inv --no-dedupe pz
    The --no-dedupe flag allows duplication of
    tasks.
    The default is to not allow.
    So in this case (default mode):
    clear--> python_zen --> sleep(5) --> clear
    Turns to:
    clear--> python_zen --> sleep(5)
    """
    c.run("python -m this")


@task(build)
def timec(c):
    print("The multiplier of C-API to Python is", timeit('geo_py.geo(0.5, 10_000)', setup='import geo_py', number=1_000) /
          timeit('cd.geo(0.5, 10_000)',  setup='import capi_demo1 as cd', number=1_000))


@task
def bint(c, format, progress_bar=True):
    """
     Checkout: inv -h bint
     and see the --[no-]progress-bar parameters
    """
    if progress_bar:
        print(format)


@task
def binf(c, format, progress_bar=False):
    if progress_bar:
        print(format)

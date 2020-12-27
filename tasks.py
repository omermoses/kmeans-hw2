from invoke import task

@task(aliases=['del'])
def delete(c):
    c.run("rm *mykmeanssp*.so")
    print("Done cleaning")


@task()
def build(c):
    c.run("python3.8.5 setup.py build_ext --inplace")
    print("Done building")
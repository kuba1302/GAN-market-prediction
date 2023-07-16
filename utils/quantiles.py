def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f

    return decorator


def q_at(y):
    @rename(f"q{y:0.2f}")
    def q(x):
        return x.quantile(y)

    return q

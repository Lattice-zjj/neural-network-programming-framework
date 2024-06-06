import sys
import numdifftools as nd

import numpy as np
import mugrade
import nnwf as ndl



def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [x.numpy() for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(args))
    )
    assert error < tol
    return computed_grads



def test_topo_sort():
    # Test case 1
    a1, b1 = ndl.Tensor(np.asarray([[0.88282157]])), ndl.Tensor(np.asarray([[0.90170084]]))
    c1 = 3*a1*a1 + 4*b1*a1 - a1

    soln = np.array([np.array([[0.88282157]]),
                     np.array([[2.64846471]]),
                     np.array([[2.33812177]]),
                     np.array([[0.90170084]]),
                     np.array([[3.60680336]]),
                     np.array([[3.1841638]]),
                     np.array([[5.52228558]]),
                     np.array([[-0.88282157]]),
                     np.array([[4.63946401]])])

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([c1])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

    # Test case 2
    a1, b1 = ndl.Tensor(np.asarray([[0.20914675], [0.65264178]])), ndl.Tensor(np.asarray([[0.65394286, 0.08218317]]))
    c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

    soln = [np.array([[0.65394286, 0.08218317]]),
            np.array([[0.20914675], [0.65264178]]),
            np.array([[0.19040619]]),
            np.array([[1.53101102, 0.19240724]]),
            np.array([[0.44577898]]), np.array([[0.63618518]]),
            np.array([[1.90855553]]), np.array([[3.40855553]])]

    topo_order = [x.numpy() for x in ndl.autograd.find_topo_sort([c1])]

    assert len(soln) == len(topo_order)
    # step through list as entries differ in length
    for t, s in zip(topo_order, soln):
        np.testing.assert_allclose(t, s, rtol=1e-06, atol=1e-06)

    # Test case 3
    a = ndl.Tensor(np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]))
    b = ndl.Tensor(np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]))
    e = (a@b + b - a)@a

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([e])])

    soln = np.array([np.array([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),
                     np.array([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]),
                     np.array([[1.6252339, -1.38248184], [1.25355725, -0.03148146]]),
                     np.array([[2.97095081, -2.33832617], [0.25927152, -0.07165645]]),
                     np.array([[-1.4335016, -0.30559972], [-0.08130171, 1.15072371]]),
                     np.array([[1.53744921, -2.64392589], [0.17796981, 1.07906726]]),
                     np.array([[1.98898021, 3.51227226], [0.34285002, -1.18732075]])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)




def test_compute_gradient():
    gradient_check(lambda A,B,C : ndl.summation((A@B+C)*(A@B), axes=None),
                   ndl.Tensor(np.random.randn(10,9)),
                   ndl.Tensor(np.random.randn(9,8)),
                   ndl.Tensor(np.random.randn(10,8)), backward=True)
    gradient_check(lambda A,B : ndl.summation(ndl.broadcast_to(A,shape=(10,9))*B, axes=None),
                   ndl.Tensor(np.random.randn(10,1)),
                   ndl.Tensor(np.random.randn(10,9)), backward=True)
    gradient_check(lambda A,B,C : ndl.summation(ndl.reshape(A,shape=(10,10))@B/5+C, axes=None),
                   ndl.Tensor(np.random.randn(100)),
                   ndl.Tensor(np.random.randn(10,5)),
                   ndl.Tensor(np.random.randn(10,5)), backward=True)

    # check gradient of gradient
    x2 = ndl.Tensor([6])
    x3 = ndl.Tensor([0])
    y = x2 * x2 + x2 * x3
    y.backward()
    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()
    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad
    x2_val = x2.numpy()
    x3_val = x3.numpy()
    assert y.numpy() == x2_val * x2_val + x2_val * x3_val
    assert grad_x2.numpy() == 2 * x2_val + x3_val
    assert grad_x3.numpy() == x2_val
    assert grad_x2_x2.numpy() == 2
    assert grad_x2_x3.numpy() == 1

import matplotlib.pyplot as plt
import numpy as np
import math

#-----------------------------------------------------------------------------------------------------
#------------------------------------techniques-------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
phi = (-1 + 5**.5)/2

def gss(f, x0, x3,eps=1e-3):
    count = 0
    x1 = (1 - phi) * (x3 - x0) + x0
    x2 = (phi * (x3 - x0)) + x0

    f0, f1, f2, f3 = [f(x) for x in [x0, x1, x2, x3]]
    count+=4

    while (abs(x3 - x0)>2*eps):
        if f0 < f1 and f0 < f2 and f0 < f3:
            x3 = x2
            f3 = f2
            x2 = x1
            f2 = f1
            x1 = (1 - phi) * (x3 - x0) + x0
            f1 = f(x1)
            count+=1
        elif f3 < f2 and f3 < f1 and f3 < f0:
            x0 = x1
            f0 = f1
            x1 = x2
            f1 = f2
            x2 = (phi * (x3 - x0)) + x0
            f2 = f(x2)
            count+=1
        if f1 < f2:
            x3 = x2
            f3 = f2
            x2 = x1
            f2 = f1
            x1 = (1 - phi) * (x3 - x0) + x0
            f1 = f(x1)
            count+=1
        else:
            x0 = x1
            f0 = f1
            x1 = x2
            f1 = f2
            x2 = (phi * (x3 - x0)) + x0
            f2 = f(x2)
            count+=1
    min_idx = sorted([0,1,2,3],key=lambda x: [f0,f1,f2,f3][x])[0]
    min_x = [x0,x1,x2,x3][min_idx]
    min_f = [f0,f1,f2,f3][min_idx]
    return {"x":min_x,"fun":min_f,"count":count}

#-----------------------------------------------------------------------------------------------------
#-----------------------------------functions---------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
# rosenbrock, ackley, sum of different powers, sphere, dixon price, levy

# Rosenbrock function
def rosenbrock2D(x,y):
    return (1 - x)**2.0 + 100*(y - x**2.0)**2.0

def rosenbrockND(xvec):
    dim = xvec.shape[0]
    result = 0
    for i in range(dim-1):
        result += rosenbrock2D(xvec[i], xvec[i+1])

    return result

# Ackley function
# f(0,0,...,0,0) = 0
def ackleyND(xvec):
    dim = xvec.shape[0]
    a = 20
    b = 0.2
    c = 2*math.pi
    fracd = 1/dim
    first = 0
    for i in range(dim):
        first+=(xvec[i]**2)
    second = 0
    for i in range(dim):
        second+=(math.cos(c*xvec[i]))
    return -a*math.exp(-b*math.sqrt(fracd*first))-math.exp(fracd*second)+a+math.exp(1)

# Sum of Different Powers Function
# f(0,0,...,0,0) = 0
def sodpND(xvec):
    dim = xvec.shape[0]
    result = 0
    for i in range(dim):
        result+=abs(xvec[i])**(i+1)
    return result

# sphere function
# f(0,0,0,...,0,0) = 0
def sphereND(xvec):
    dim = xvec.shape[0]
    result = 0
    for i in range(dim):
        result+= (xvec[i])**2
    return result

# Dixon-Price function
def dixonprice2D(x,y,i):
    return i*(2*x**2-y)**2

# f(x*) = 0 at xi = 2^((-2^i-2)/2^i) for i = 1,...,d
def dixonpriceND(xvec):
    dim = xvec.shape[0]
    start = (xvec[0]-1)**2
    sum = 0
    for i in range(1,dim):
        sum += dixonprice2D(xvec[i],xvec[i-1],i)
    return start+sum

# Levy function
def levy2D(x,i):
    wi = 1+(x[i]-1)/4
    return ((wi-1)**2)*(1+10*(math.sin(math.pi*wi+1))**2)

# f(1,1,1,...,1,1,1) = 0
def levyND(xvec):
    dim = xvec.shape[0]
    w1 = 1+((xvec[0]-1)/4)
    start = math.sin(math.pi*w1)**2
    wd = 1+(xvec[dim-1]-1)/4
    mid = 0
    for i in range(dim-1):
        mid+=levy2D(xvec,i)
    end = ((wd-1)**2)*(1+(math.sin(math.pi*wd+1))**2)
    return start+mid+end

#-----------------------------------------------------------------------------------------------------
#-----------------------------------optimization algorithms-------------------------------------------
#-----------------------------------------------------------------------------------------------------

# gradient finite difference
def grad_fd(f, x0, h=1e-6):
    grad = np.zeros(x0.shape)
    count = 0
    for i in range(grad.shape[0]):
        axis = np.zeros(x0.shape)
        axis[i] = h
        grad[i] = (f(x0 + axis) - f(x0))/h
        count+=2
    return grad,count

# base line search implementation, not actually used in this file
def line_search(f, x0, eps=1e-6):

    # one cycle of coordinate descent:
    x_old = np.inf*x0.copy()

    while np.linalg.norm(x0 - x_old) > eps:
        x_old = x0.copy()
        direction = grad_fd(f, x0)
        myfunc = lambda var: f(x0 + var*direction)
        result = gss(myfunc, -10, 10)
        x0 = x0 + result['x']*direction

    return {'x':x0, 'fun':f(x0)}

# base coordinate descent implementation, not actually used in this file
def coord_desc(f, x0, eps=1e-6):

    # one cycle of coordinate descent:
    x_old = np.inf*x0.copy()

    while np.linalg.norm(x0 - x_old) > eps:
        x_old = x0.copy()
        for i in range(x0.shape[0]):
            direction = np.zeros(x0.shape)
            direction[i] = 1
            myfunc = lambda var: f(x0 + var*direction)
            result = gss(myfunc, -10, 10)
            x0[i] = x0[i] + result['x']
    return {'x':x0,'fun':f(x0)}


# gss, coord desc, cyclic, limited iterations
def cd_cyc_base(f, x0, eps=1e-6):

    xpts = []
    count = 0
    dim = x0.shape[0]

    # one cycle of coordinate descent:
    x_old = np.inf*x0.copy()
    cyc_count = 0
    while cyc_count<1e4:
    # while np.linalg.norm(x0 - x_old) > 2*eps:
        x_old = x0.copy()
        for i in range(dim):
            direction = np.zeros(x0.shape)
            direction[i] = 1
            myfunc = lambda var: f(x0 + var*direction) # [item * var for item in direction])
            result = gss(myfunc,-3, 3)
            x0[i] = x0[i] + result['x']
            count+=result['count']
            xpts.append(x0.copy())
        cyc_count+=1
    return x0, xpts, count

# gss, coord desc, random cyclic, limited iterations
def cd_rand_new(f, x0, eps=1e-6):

    xpts = []
    count = 0

    # one cycle of coordinate descent:
    x_old = np.inf*x0.copy()
    cyc_count = 0

    while cyc_count<1e4: # np.linalg.norm(x0 - x_old) > eps:
        x_old = x0.copy()
        np.random.shuffle(x0)
        for i in range(x0.shape[0]):
            direction = np.zeros(x0.shape)
            direction[i] = 1
            myfunc = lambda var: f(x0 + var*direction)
            result = gss(myfunc, -10, 10)
            x0[i] = x0[i] + result['x']
            count+=result['count']
            xpts.append(x0.copy())
        cyc_count+=1
    return x0, xpts, count

# gss, line search, gss determines step size, gradient estimated using the (f(x0 + axis) - f(x0))/h equation, terminates when x-xnew is small enough
def ls_grad_base(f, x0, eps):

    xpts = []
    count = 0

    # one cycle of coordinate descent:
    x_old = np.inf*x0.copy()

    while np.linalg.norm(x0 - x_old) > eps:
        x_old = x0.copy()
        direction,grad_count = grad_fd(f, x0)
        count+=grad_count
        myfunc = lambda var: f(x0 + var*direction)
        result = gss(myfunc, -10, 10)
        x0 = x0 + result['x']*direction
        count+=result['count']
        xpts.append(x0.copy())
    
    return x0, xpts, count

# gss, gradient descent, step 0.0001, gradient estimated using the (f(x0 + axis) - f(x0))/h equation, terminates when x-xnew is small enough
def gd_base(f, x0, eps, lr=.0001):

    xpts = []
    count = 0

    # one cycle of coordinate descent:
    x_old = np.inf*x0.copy()

    while np.linalg.norm(x0 - x_old) > eps:
        x_old = x0.copy()
        direction,grad_count = grad_fd(f, x0)
        count+=grad_count
        x0 = x0 - lr*direction
        xpts.append(x0.copy())

    return x0, xpts, count 

#-----------------------------------------------------------------------------------------------------
#----------------------------------display------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

# 3 nested, 1 for function, 1 for algorithms, 1 for epsilons
test_functions = [rosenbrockND,ackleyND,sodpND,sphereND,dixonpriceND,levyND]
opt_methods = [cd_cyc_base,cd_rand_new,ls_grad_base,gd_base]
eps_vals= np.arange(1e-3,1e-2,1e-3)

fig,ax = plt.subplots(len(test_functions),1)

for i in range(len(test_functions)):
    # change the 2 here to change the dimension of the matrix plugged into the experiments
    random_start = 10*np.random.random((2,))
    for j in range(len(opt_methods)):
        results = []
        name = ""+opt_methods[j].__name__+","+test_functions[i].__name__+""
        print(name)
        for eps in eps_vals:
            result = opt_methods[j](test_functions[i],random_start,eps)
            # needs to be the function evaluation count, not the actual number
            results.append(result[2])
        plt.yscale('log')
        ax[i].scatter(eps_vals,results,label=name)
    ax[i].legend(loc='center right')
plt.show()

# rosenbrock, ackley, sum of different powers, sphere, dixon price, levy

# gss, coord desc, cyclic, limited iterations
# gss, coord desc, random cyclic, limited iterations
# gss, line search, gss determines step size, gradient estimated using the (f(x0 + axis) - f(x0))/h equation, terminates when x-xnew is small enough
# gss, gradient descent, step 0.0001, gradient estimated using the (f(x0 + axis) - f(x0))/h equation, terminates when x-xnew is small enough

# 6 different plots with each algorithm's results from running on the plot's function

# Analysis:
# I ended up having to switch my first two algorithm's termination conditions to limited iterations
# rather than changes in x because my gss function is broken for some reason that I couldn't figure out
# and it was just running for several minutes and not actually improving, sometimes even jumping out
# to 1e+25 or more
# however, my line search and gradient descent both actually get more efficient as the epsilon decreases
# more specifically, it seems like they sharply increase in efficiency then flatten out
# this would probably be easier to see if you cut out the two coordinate descent algorithms since they
# both throw off the scale by taking around 480,000 function calls in the 1,000 or so iterations I let
# them have.
def rk4(x, ode, h=0.1):
  k1 = ode(x)
  k2 = ode(x+h/2*k1)
  k3 = ode(x+h/2*k2)
  k4 = ode(x+h*k3)
  return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
